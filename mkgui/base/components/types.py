import base64
from typing import Any, Dict, overload


class FileContent(str):
    def as_bytes(self) -> bytes:
        return base64.b64decode(self, validate=True)

    def as_str(self) -> str:
        return self.as_bytes().decode()

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(format="byte")

    @classmethod
    def __get_validators__(cls) -> Any:  # type: ignore
        yield cls.validate

    @classmethod
    def validate(cls, value: Any) -> "FileContent":
        if isinstance(value, FileContent):
            return value
        elif isinstance(value, str):
            return FileContent(value)
        elif isinstance(value, (bytes, bytearray, memoryview)):
            return FileContent(base64.b64encode(value).decode())
        else:
            raise Exception("Wrong type")

# # 暂时无法使用，因为浏览器中没有考虑选择文件夹
# class DirectoryContent(FileContent):
#     @classmethod
#     def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
#         field_schema.update(format="path")

#     @classmethod
#     def validate(cls, value: Any) -> "DirectoryContent":
#         if isinstance(value, DirectoryContent):
#             return value
#         elif isinstance(value, str):
#             return DirectoryContent(value)
#         elif isinstance(value, (bytes, bytearray, memoryview)):
#             return DirectoryContent(base64.b64encode(value).decode())
#         else:
#             raise Exception("Wrong type")
