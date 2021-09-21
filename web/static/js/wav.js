/*
wav编码器+编码引擎
https://github.com/xiangyuecn/Recorder

当然最佳推荐使用mp3、wav格式，代码也是优先照顾这两种格式
浏览器支持情况
https://developer.mozilla.org/en-US/docs/Web/HTML/Supported_media_formats

编码原理：给pcm数据加上一个44直接的wav头即成wav文件；pcm数据就是Recorder中的buffers原始数据（重新采样），16位时为LE小端模式（Little Endian），实质上是未经过任何编码处理
*/
(function(){
"use strict";

Recorder.prototype.enc_wav={
	stable:true
	,testmsg:"支持位数8位、16位（填在比特率里面），采样率取值无限制"
};
Recorder.prototype.wav=function(res,True,False){
		var This=this,set=This.set
			,size=res.length
			,sampleRate=set.sampleRate
			,bitRate=set.bitRate==8?8:16;
		
		//编码数据 https://github.com/mattdiamond/Recorderjs https://www.cnblogs.com/blqw/p/3782420.html https://www.cnblogs.com/xiaoqi/p/6993912.html
		var dataLength=size*(bitRate/8);
		var buffer=new ArrayBuffer(44+dataLength);
		var data=new DataView(buffer);
		
		var offset=0;
		var writeString=function(str){
			for (var i=0;i<str.length;i++,offset++) {
				data.setUint8(offset,str.charCodeAt(i));
			};
		};
		var write16=function(v){
			data.setUint16(offset,v,true);
			offset+=2;
		};
		var write32=function(v){
			data.setUint32(offset,v,true);
			offset+=4;
		};
		
		/* RIFF identifier */
		writeString('RIFF');
		/* RIFF chunk length */
		write32(36+dataLength);
		/* RIFF type */
		writeString('WAVE');
		/* format chunk identifier */
		writeString('fmt ');
		/* format chunk length */
		write32(16);
		/* sample format (raw) */
		write16(1);
		/* channel count */
		write16(1);
		/* sample rate */
		write32(sampleRate);
		/* byte rate (sample rate * block align) */
		write32(sampleRate*(bitRate/8));
		/* block align (channel count * bytes per sample) */
		write16(bitRate/8);
		/* bits per sample */
		write16(bitRate);
		/* data chunk identifier */
		writeString('data');
		/* data chunk length */
		write32(dataLength);
		// 写入采样数据
		if(bitRate==8) {
			for(var i=0;i<size;i++,offset++) {
				//16转8据说是雷霄骅的 https://blog.csdn.net/sevennight1989/article/details/85376149 细节比blqw的按比例的算法清晰点，虽然都有明显杂音
				var val=(res[i]>>8)+128;
				data.setInt8(offset,val,true);
			};
		}else{
			for (var i=0;i<size;i++,offset+=2){
				data.setInt16(offset,res[i],true);
			};
		};
		
		
		True(new Blob([data.buffer],{type:"audio/wav"}));
	}
})();