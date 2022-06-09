#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
  Filename: digit_2_chinese.py
  Desc    :
  Author  : nfyn
  Created : 2022/5/22
-------------------------------------------------
"""
import re
from typing import Union

HAN_STR: str = '零一二三四五六七八九'


def four_digit_2_chinese(num_str: str, unit: str) -> str:
    """
    每4位数字进行转换成中文处理函数
    :param num_str: 数字字符串，长度为4
    :param unit: 单位
    :return: 中文数字+单位
    """
    unit_lst: list = ['', '十', '百', '千']
    result: str = ''
    for idx, num in enumerate(num_str):
        if num != '0':
            # 如果数字不是0，数字转换为汉字，拼接上单位
            try:
                result = HAN_STR[int(num)] + unit_lst[idx] + result
            except ValueError as e:
                # 可处理非数字字符串，直接按原值返回
                result = num + result
        else:
            result = '零' + result

    # 处理`零`的显示
    if all(i == '零' for i in result):
        # 如果在和单位拼接前，`全是零`，则返回空字符串
        return ''
    else:
        # 如果在和单位拼接前，`不全是零`，对`连续两个零`进行替换为`零`，对于`右侧一个零`的情况，直接删除`右侧的零`
        rep: str = re.sub(r'零{2,}', '零', result).rstrip('零')
        if rep.startswith('一十'):
            # 如果以`一十`开头，直接删除`左侧的一`，再和单位进行拼接返回
            return rep.lstrip('一') + unit
        else:
            # 和单位进行拼接返回
            return rep + unit


def integer_2_chinese(num_str: str) -> str:
    """
    整数部分转换成中文处理函数
    :param num_str: 整数部分的数字字符串
    :return: 整数部分的中文字符串
    """
    if all(i == '0' for i in num_str):
        # 如果为数字全为`0`，直接返回`零`
        return '零'
    # 反转数字顺序，便于从小到大对应单位
    num_reverse: str = num_str.lstrip('0')[::-1]
    # 每4位进行转换，分别对应单位是 ``, `万`10^4, `亿`10^8, `兆`10^12, `京`10^16, `垓`10^20,`秭`10^24, `穰`10^28,`沟`10^32, `涧`10^36
    unit_lst: list = ['', '万', '亿', '兆', '京', '垓', '秭', '穰', '沟', '涧', '正', '载', '极', '恒河沙', '阿僧祗', '那由他',
                      '不可思议', '无量大海', '大数']

    res: str = ''.join(
        [four_digit_2_chinese(num_reverse[idx * 4:(idx + 1) * 4], unit=unit) for idx, unit in enumerate(unit_lst)][::-1])
    return res


def decimal_2_chinese(num_str: str) -> str:
    """
    小数部分转换中文处理函数
    :param num_str: 小数部分数字字符串
    :return: 小数部分的中文字符串
    """
    result: str = ''
    for idx, num in enumerate(num_str):
        try:
            result += HAN_STR[int(num)]
        except ValueError as e:
            # 可处理非数字字符串，直接按原值返回
            result += num
    return result


def date_2_chinese(num_str: str) -> str:
    """
    日期部分转换中文处理函数
    :param num_str: 日期部分数字字符串
    :return: 日期部分的中文字符串
    """
    if isinstance(num_str, re.Match):
        num_str = num_str.group(0)

    result: str = ''
    num_lst = num_str.split('年')

    for idx, num in enumerate(num_lst[0]):
        try:
            result += HAN_STR[int(num)]
        except ValueError as e:
            # 可处理非数字字符串，直接按原值返回
            result += num
    return result + '年' + num_lst[1]


def time_2_chinese(num_str: str) -> str:
    """
    时间部分转换中文处理函数
    :param num_str: 包含时间部分数字字符串
    :return: 包含时间部分的中文字符串
    """
    if isinstance(num_str, re.Match):
        num_str = num_str.group(0)

    num_lst = num_str.split(':')
    return ''.join([item[0]+item[1] for item in zip(num_lst, ('时', '分', '秒'))])


def to_chinese(num: Union[str, int]) -> str:
    """
    数字转换成中文入口函数
    :param num: 数字或者数字字符串
    :return:完整的中文数字字符串
    """
    if isinstance(num, re.Match):
        num = num.group(0)

    num_str = str(num)
    # 分割整数和小数部分
    num_lst = num_str.split('.')
    if len(num_lst) == 1:
        # 如果只要整数部分
        return integer_2_chinese(num_lst[0])
    elif len(num_lst) == 2:
        # 如果有小数部分
        integer_str = integer_2_chinese(num_lst[0])
        decimal_str = decimal_2_chinese(num_lst[1])
        return integer_str + f'点{decimal_str}' if decimal_str else decimal_str
    else:
        raise ValueError('The number string format is not correct')


def text_to_chinese(text: str) -> str:
    """
    将文本中的数字替换成汉字
    :param text: 输入文本
    :return: 输出全部转换文汉字的文本
    """
    # 转换文本中的日期信息
    text = re.sub(r'(\d{4}).*?年.*?月', date_2_chinese, str(text))

    # 转换文本中的时间信息
    text = text.replace('：', ':')
    text = re.sub(r'(\d{1,2}).*?:.*?\d{1,2}.*?:.*?\d{1,2}', time_2_chinese, text)
    text = re.sub(r'(\d{1,2}).*?:.*?\d{1,2}.*?', time_2_chinese, text)

    # 转换文本中
    text = re.sub(r'\d(?:[\d.]*\d+)?', to_chinese, text)
    return text


if __name__ == '__main__':
    text = '好像1980年12月13日24：00:12，2023年05月30日共计303年, 300233.45'
    print(text_to_chinese(text))
