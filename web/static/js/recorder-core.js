/*
录音
https://github.com/xiangyuecn/Recorder
*/
(function(factory){
	factory(window);
	//umd returnExports.js
	if(typeof(define)=='function' && define.amd){
		define(function(){
			return Recorder;
		});
	};
	if(typeof(module)=='object' && module.exports){
		module.exports=Recorder;
	};
}(function(window){
"use strict";

//兼容环境
var LM="2021-08-03 20:01:03";
var NOOP=function(){};
//end 兼容环境 ****从以下开始copy源码*****

var Recorder=function(set){
	return new initFn(set);
};
//是否已经打开了全局的麦克风录音，所有工作都已经准备好了，就等接收音频数据了
Recorder.IsOpen=function(){
	var stream=Recorder.Stream;
	if(stream){
		var tracks=stream.getTracks&&stream.getTracks()||stream.audioTracks||[];
		var track=tracks[0];
		if(track){
			var state=track.readyState;
			return state=="live"||state==track.LIVE;
		};
	};
	return false;
};
/*H5录音时的AudioContext缓冲大小。会影响H5录音时的onProcess调用速率，相对于AudioContext.sampleRate=48000时，4096接近12帧/s，调节此参数可生成比较流畅的回调动画。
	取值256, 512, 1024, 2048, 4096, 8192, or 16384
	注意，取值不能过低，2048开始不同浏览器可能回调速率跟不上造成音质问题。
	一般无需调整，调整后需要先close掉已打开的录音，再open时才会生效。
*/
Recorder.BufferSize=4096;
//销毁已持有的所有全局资源，当要彻底移除Recorder时需要显式的调用此方法
Recorder.Destroy=function(){
	CLog("Recorder Destroy");
	Disconnect();//断开可能存在的全局Stream、资源
	
	for(var k in DestroyList){
		DestroyList[k]();
	};
};
var DestroyList={};
//登记一个需要销毁全局资源的处理方法
Recorder.BindDestroy=function(key,call){
	DestroyList[key]=call;
};
//判断浏览器是否支持录音，随时可以调用。注意：仅仅是检测浏览器支持情况，不会判断和调起用户授权，不会判断是否支持特定格式录音。
Recorder.Support=function(){
	var AC=window.AudioContext;
	if(!AC){
		AC=window.webkitAudioContext;
	};
	if(!AC){
		return false;
	};
	var scope=navigator.mediaDevices||{};
	if(!scope.getUserMedia){
		scope=navigator;
		scope.getUserMedia||(scope.getUserMedia=scope.webkitGetUserMedia||scope.mozGetUserMedia||scope.msGetUserMedia);
	};
	if(!scope.getUserMedia){
		return false;
	};
	
	Recorder.Scope=scope;
	if(!Recorder.Ctx||Recorder.Ctx.state=="closed"){
		//不能反复构造，低版本number of hardware contexts reached maximum (6)
		Recorder.Ctx=new AC();
		
		Recorder.BindDestroy("Ctx",function(){
			var ctx=Recorder.Ctx;
			if(ctx&&ctx.close){//能关掉就关掉，关不掉就保留着
				ctx.close();
				Recorder.Ctx=0;
			};
		});
	};
	return true;
};
/*初始化H5音频采集连接。如果自行提供了sourceStream将只进行一次简单的连接处理。如果是普通麦克风录音，此时的Stream是全局的，Safari上断开后就无法再次进行连接使用，表现为静音，因此使用全部使用全局处理避免调用到disconnect；全局处理也有利于屏蔽底层细节，start时无需再调用底层接口，提升兼容、可靠性。*/
var Connect=function(streamStore){
	streamStore=streamStore||Recorder;
	var bufferSize=streamStore.BufferSize||Recorder.BufferSize;
	
	var ctx=Recorder.Ctx,stream=streamStore.Stream;
	var media=stream._m=ctx.createMediaStreamSource(stream);
	var process=stream._p=(ctx.createScriptProcessor||ctx.createJavaScriptNode).call(ctx,bufferSize,1,1);//单声道，省的数据处理复杂
	
	media.connect(process);
	process.connect(ctx.destination);
	
	var calls=stream._call;
	process.onaudioprocess=function(e){
		for(var k0 in calls){//has item
			var o=e.inputBuffer.getChannelData(0);//块是共享的，必须复制出来
			var size=o.length;
			
			var pcm=new Int16Array(size);
			var sum=0;
			for(var j=0;j<size;j++){//floatTo16BitPCM 
				var s=Math.max(-1,Math.min(1,o[j]));
				s=s<0?s*0x8000:s*0x7FFF;
				pcm[j]=s;
				sum+=Math.abs(s);
			};
			
			for(var k in calls){
				calls[k](pcm,sum);
			};
			
			return;
		};
	};
};
var Disconnect=function(streamStore){
	streamStore=streamStore||Recorder;
	var isGlobal=streamStore==Recorder;
	
	var stream=streamStore.Stream;
	if(stream){
		if(stream._m){
			stream._m.disconnect();
			stream._p.disconnect();
			stream._p.onaudioprocess=stream._p=stream._m=null;
		};
		
		if(isGlobal){//全局的时候，要把流关掉（麦克风），直接提供的流不处理
			var tracks=stream.getTracks&&stream.getTracks()||stream.audioTracks||[];
			for(var i=0;i<tracks.length;i++){
				var track=tracks[i];
				track.stop&&track.stop();
			};
			stream.stop&&stream.stop();
		};
	};
	streamStore.Stream=0;
};

/*对pcm数据的采样率进行转换
pcmDatas: [[Int16,...]] pcm片段列表
pcmSampleRate:48000 pcm数据的采样率
newSampleRate:16000 需要转换成的采样率，newSampleRate>=pcmSampleRate时不会进行任何处理，小于时会进行重新采样
prevChunkInfo:{} 可选，上次调用时的返回值，用于连续转换，本次调用将从上次结束位置开始进行处理。或可自行定义一个ChunkInfo从pcmDatas指定的位置开始进行转换
option:{ 可选，配置项
		frameSize:123456 帧大小，每帧的PCM Int16的数量，采样率转换后的pcm长度为frameSize的整数倍，用于连续转换。目前仅在mp3格式时才有用，frameSize取值为1152，这样编码出来的mp3时长和pcm的时长完全一致，否则会因为mp3最后一帧录音不够填满时添加填充数据导致mp3的时长变长。
		frameType:"" 帧类型，一般为rec.set.type，提供此参数时无需提供frameSize，会自动使用最佳的值给frameSize赋值，目前仅支持mp3=1152(MPEG1 Layer3的每帧采采样数)，其他类型=1。
			以上两个参数用于连续转换时使用，最多使用一个，不提供时不进行帧的特殊处理，提供时必须同时提供prevChunkInfo才有作用。最后一段数据处理时无需提供帧大小以便输出最后一丁点残留数据。
	}

返回ChunkInfo:{
	//可定义，从指定位置开始转换到结尾
	index:0 pcmDatas已处理到的索引
	offset:0.0 已处理到的index对应的pcm中的偏移的下一个位置
	
	//仅作为返回值
	frameNext:null||[Int16,...] 下一帧的部分数据，frameSize设置了的时候才可能会有
	sampleRate:16000 结果的采样率，<=newSampleRate
	data:[Int16,...] 转换后的PCM结果；如果是连续转换，并且pcmDatas中并没有新数据时，data的长度可能为0
}
*/
Recorder.SampleData=function(pcmDatas,pcmSampleRate,newSampleRate,prevChunkInfo,option){
	prevChunkInfo||(prevChunkInfo={});
	var index=prevChunkInfo.index||0;
	var offset=prevChunkInfo.offset||0;
	
	var frameNext=prevChunkInfo.frameNext||[];
	option||(option={});
	var frameSize=option.frameSize||1;
	if(option.frameType){
		frameSize=option.frameType=="mp3"?1152:1;
	};
	
	var size=0;
	for(var i=index;i<pcmDatas.length;i++){
		size+=pcmDatas[i].length;
	};
	size=Math.max(0,size-Math.floor(offset));
	
	//采样 https://www.cnblogs.com/blqw/p/3782420.html
	var step=pcmSampleRate/newSampleRate;
	if(step>1){//新采样低于录音采样，进行抽样
		size=Math.floor(size/step);
	}else{//新采样高于录音采样不处理，省去了插值处理
		step=1;
		newSampleRate=pcmSampleRate;
	};
	
	size+=frameNext.length;
	var res=new Int16Array(size);
	var idx=0;
	//添加上一次不够一帧的剩余数据
	for(var i=0;i<frameNext.length;i++){
		res[idx]=frameNext[i];
		idx++;
	};
	//处理数据
	for (var nl=pcmDatas.length;index<nl;index++) {
		var o=pcmDatas[index];
		var i=offset,il=o.length;
		while(i<il){
			//res[idx]=o[Math.round(i)]; 直接简单抽样
			
			//https://www.cnblogs.com/xiaoqi/p/6993912.html
			//当前点=当前点+到后面一个点之间的增量，音质比直接简单抽样好些
			var before = Math.floor(i);
			var after = Math.ceil(i);
			var atPoint = i - before;
			
			var beforeVal=o[before];
			var afterVal=after<il ? o[after]
				: (//后个点越界了，查找下一个数组
					(pcmDatas[index+1]||[beforeVal])[0]||0
				);
			res[idx]=beforeVal+(afterVal-beforeVal)*atPoint;
			
			idx++;
			i+=step;//抽样
		};
		offset=i-il;
	};
	//帧处理
	frameNext=null;
	var frameNextSize=res.length%frameSize;
	if(frameNextSize>0){
		var u8Pos=(res.length-frameNextSize)*2;
		frameNext=new Int16Array(res.buffer.slice(u8Pos));
		res=new Int16Array(res.buffer.slice(0,u8Pos));
	};
	
	return {
		index:index
		,offset:offset
		
		,frameNext:frameNext
		,sampleRate:newSampleRate
		,data:res
	};
};


/*计算音量百分比的一个方法
pcmAbsSum: pcm Int16所有采样的绝对值的和
pcmLength: pcm长度
返回值：0-100，主要当做百分比用
注意：这个不是分贝，因此没用volume当做名称*/
Recorder.PowerLevel=function(pcmAbsSum,pcmLength){
	/*计算音量 https://blog.csdn.net/jody1989/article/details/73480259
	更高灵敏度算法:
		限定最大感应值10000
			线性曲线：低音量不友好
				power/10000*100 
			对数曲线：低音量友好，但需限定最低感应值
				(1+Math.log10(power/10000))*100
	*/
	var power=(pcmAbsSum/pcmLength) || 0;//NaN
	var level;
	if(power<1251){//1250的结果10%，更小的音量采用线性取值
		level=Math.round(power/1250*10);
	}else{
		level=Math.round(Math.min(100,Math.max(0,(1+Math.log(power/10000)/Math.log(10))*100)));
	};
	return level;
};




//带时间的日志输出，CLog(msg,errOrLogMsg, logMsg...) err为数字时代表日志类型1:error 2:log默认 3:warn，否则当做内容输出，第一个参数不能是对象因为要拼接时间，后面可以接无数个输出参数
var CLog=function(msg,err){
	var now=new Date();
	var t=("0"+now.getMinutes()).substr(-2)
		+":"+("0"+now.getSeconds()).substr(-2)
		+"."+("00"+now.getMilliseconds()).substr(-3);
	var arr=["["+t+" Recorder]"+msg];
	var a=arguments;
	var i=2,fn=console.log;
	if(typeof(err)=="number"){
		fn=err==1?console.error:err==3?console.warn:fn;
	}else{
		i=1;
	};
	for(;i<a.length;i++){
		arr.push(a[i]);
	};
	fn.apply(console,arr);
};
Recorder.CLog=CLog;




var ID=0;
function initFn(set){
	this.id=++ID;
	
	//如果开启了流量统计，这里将发送一个图片请求
	Recorder.Traffic&&Recorder.Traffic();
	
	
	var o={
		type:"mp3" //输出类型：mp3,wav，wav输出文件尺寸超大不推荐使用，但mp3编码支持会导致js文件超大，如果不需支持mp3可以使js文件大幅减小
		,bitRate:16 //比特率 wav:16或8位，MP3：8kbps 1k/s，8kbps 2k/s 录音文件很小
		
		,sampleRate:16000 //采样率，wav格式大小=sampleRate*时间；mp3此项对低比特率有影响，高比特率几乎无影响。
					//wav任意值，mp3取值范围：48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000
					//采样率参考https://www.cnblogs.com/devin87/p/mp3-recorder.html
		
		,onProcess:NOOP //fn(buffers,powerLevel,bufferDuration,bufferSampleRate,newBufferIdx,asyncEnd) buffers=[[Int16,...],...]：缓冲的PCM数据，为从开始录音到现在的所有pcm片段；powerLevel：当前缓冲的音量级别0-100，bufferDuration：已缓冲时长，bufferSampleRate：缓冲使用的采样率（当type支持边录边转码(Worker)时，此采样率和设置的采样率相同，否则不一定相同）；newBufferIdx:本次回调新增的buffer起始索引；asyncEnd:fn() 如果onProcess是异步的(返回值为true时)，处理完成时需要调用此回调，如果不是异步的请忽略此参数，此方法回调时必须是真异步（不能真异步时需用setTimeout包裹）。onProcess返回值：如果返回true代表开启异步模式，在某些大量运算的场合异步是必须的，必须在异步处理完成时调用asyncEnd(不能真异步时需用setTimeout包裹)，在onProcess执行后新增的buffer会全部替换成空数组，因此本回调开头应立即将newBufferIdx到本次回调结尾位置的buffer全部保存到另外一个数组内，处理完成后写回buffers中本次回调的结尾位置。
		
		//*******高级设置******
		//,sourceStream:MediaStream Object
				//可选直接提供一个媒体流，从这个流中录制、实时处理音频数据（当前Recorder实例独享此流）；不提供时为普通的麦克风录音，由getUserMedia提供音频流（所有Recorder实例共享同一个流）
				//比如：audio、video标签dom节点的captureStream方法（实验特性，不同浏览器支持程度不高）返回的流；WebRTC中的remote流；自己创建的流等
				//注意：流内必须至少存在一条音轨(Audio Track)，比如audio标签必须等待到可以开始播放后才会有音轨，否则open会失败
		
		//,audioTrackSet:{ deviceId:"",groupId:"", autoGainControl:true, echoCancellation:true, noiseSuppression:true }
				//普通麦克风录音时getUserMedia方法的audio配置参数，比如指定设备id，回声消除、降噪开关；注意：提供的任何配置值都不一定会生效
				//由于麦克风是全局共享的，所以新配置后需要close掉以前的再重新open
				//更多参考: https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackConstraints
		
		//,disableEnvInFix:false 内部参数，禁用设备卡顿时音频输入丢失补偿功能
		
		//,takeoffEncodeChunk:NOOP //fn(chunkBytes) chunkBytes=[Uint8,...]：实时编码环境下接管编码器输出，当编码器实时编码出一块有效的二进制音频数据时实时回调此方法；参数为二进制的Uint8Array，就是编码出来的音频数据片段，所有的chunkBytes拼接在一起即为完整音频。本实现的想法最初由QQ2543775048提出
				//当提供此回调方法时，将接管编码器的数据输出，编码器内部将放弃存储生成的音频数据；环境要求比较苛刻：如果当前环境不支持实时编码处理，将在open时直接走fail逻辑
				//因此提供此回调后调用stop方法将无法获得有效的音频数据，因为编码器内没有音频数据，因此stop时返回的blob将是一个字节长度为0的blob
				//目前只有mp3格式实现了实时编码，在支持实时处理的环境中将会实时的将编码出来的mp3片段通过此方法回调，所有的chunkBytes拼接到一起即为完整的mp3，此种拼接的结果比mock方法实时生成的音质更加，因为天然避免了首尾的静默
				//目前除mp3外其他格式不可以提供此回调，提供了将在open时直接走fail逻辑
	};
	
	for(var k in set){
		o[k]=set[k];
	};
	this.set=o;
	
	this._S=9;//stop同步锁，stop可以阻止open过程中还未运行的start
	this.Sync={O:9,C:9};//和Recorder.Sync一致，只不过这个是非全局的，仅用来简化代码逻辑，无实际作用
};
//同步锁，控制对Stream的竞争；用于close时中断异步的open；一个对象open如果变化了都要阻止close，Stream的控制权交个新的对象
Recorder.Sync={/*open*/O:9,/*close*/C:9};

Recorder.prototype=initFn.prototype={
	//流相关的数据存储在哪个对象里面；如果提供了sourceStream，数据直接存储在当前对象中，否则存储在全局
	_streamStore:function(){
		if(this.set.sourceStream){
			return this;
		}else{
			return Recorder;
		}
	}
	
	//打开录音资源True(),False(msg,isUserNotAllow)，需要调用close。注意：此方法是异步的；一般使用时打开，用完立即关闭；可重复调用，可用来测试是否能录音
	,open:function(True,False){
		var This=this,streamStore=This._streamStore();
		True=True||NOOP;
		var failCall=function(errMsg,isUserNotAllow){
			isUserNotAllow=!!isUserNotAllow;
			CLog("录音open失败："+errMsg+",isUserNotAllow:"+isUserNotAllow,1);
			False&&False(errMsg,isUserNotAllow);
		};
		
		var ok=function(){
			CLog("open成功");
			True();
			
			This._SO=0;//解除stop对open中的start调用的阻止
		};
		
		
		//同步锁
		var Lock=streamStore.Sync;
		var lockOpen=++Lock.O,lockClose=Lock.C;
		This._O=This._O_=lockOpen;//记住当前的open，如果变化了要阻止close，这里假定了新对象已取代当前对象并且不再使用
		This._SO=This._S;//记住open过程中的stop，中途任何stop调用后都不能继续open中的start
		var lockFail=function(){
			//允许多次open，但不允许任何一次close，或者自身已经调用了关闭
			if(lockClose!=Lock.C || !This._O){
				var err="open被取消";
				if(lockOpen==Lock.O){
					//无新的open，已经调用了close进行取消，此处应让上次的close明确生效
					This.close();
				}else{
					err="open被中断";
				};
				failCall(err);
				return true;
			};
		};
		
		//环境配置检查
		var checkMsg=This.envCheck({envName:"H5",canProcess:true});
		if(checkMsg){
			failCall("不能录音："+checkMsg);
			return;
		};
		
		
		//***********已直接提供了音频流************
		if(This.set.sourceStream){
			if(!Recorder.Support()){
				failCall("不支持此浏览器从流中获取录音");
				return;
			};
			
			Disconnect(streamStore);//可能已open过，直接先尝试断开
			This.Stream=This.set.sourceStream;
			This.Stream._call={};
			
			try{
				Connect(streamStore);
			}catch(e){
				failCall("从流中打开录音失败："+e.message);
				return;
			}
			ok();
			return;
		};
		
		
		//***********打开麦克风得到全局的音频流************
		var codeFail=function(code,msg){
			try{//跨域的优先检测一下
				window.top.a;
			}catch(e){
				failCall('无权录音(跨域，请尝试给iframe添加麦克风访问策略，如allow="camera;microphone")');
				return;
			};
			
			if(/Permission|Allow/i.test(code)){
				failCall("用户拒绝了录音权限",true);
			}else if(window.isSecureContext===false){
				failCall("无权录音(需https)");
			}else if(/Found/i.test(code)){//可能是非安全环境导致的没有设备
				failCall(msg+"，无可用麦克风");
			}else{
				failCall(msg);
			};
		};
		
		
		//如果已打开并且有效就不要再打开了
		if(Recorder.IsOpen()){
			ok();
			return;
		};
		if(!Recorder.Support()){
			codeFail("","此浏览器不支持录音");
			return;
		};
				
		//请求权限，如果从未授权，一般浏览器会弹出权限请求弹框
		var f1=function(stream){
			Recorder.Stream=stream;
			stream._call={};//此时is open，但并未connect，是允许绑定接收数据的
			if(lockFail())return;
			
			//https://github.com/xiangyuecn/Recorder/issues/14 获取到的track.readyState!="live"，刚刚回调时可能是正常的，但过一下可能就被关掉了，原因不明。延迟一下保证真异步。对正常浏览器不影响
			setTimeout(function(){
				if(lockFail())return;
				
				if(Recorder.IsOpen()){
					Connect();
					ok();
				}else{
					failCall("录音功能无效：无音频流");
				};
			},100);
		};
		var f2=function(e){
			var code=e.name||e.message||e.code+":"+e;
			CLog("请求录音权限错误",1,e);
			
			codeFail(code,"无法录音："+code);
		};
		var pro=Recorder.Scope.getUserMedia({audio:This.set.audioTrackSet||true},f1,f2);
		if(pro&&pro.then){
			pro.then(f1)[True&&"catch"](f2); //fix 关键字，保证catch压缩时保持字符串形式
		};
	}
	//关闭释放录音资源
	,close:function(call){
		call=call||NOOP;
		
		var This=this,streamStore=This._streamStore();
		This._stop();
		
		var Lock=streamStore.Sync;
		This._O=0;
		if(This._O_!=Lock.O){
			//唯一资源Stream的控制权已交给新对象，这里不能关闭。此处在每次都弹权限的浏览器内可能存在泄漏，新对象被拒绝权限可能不会调用close，忽略这种不处理
			CLog("close被忽略",3);
			call();
			return;
		};
		Lock.C++;//获得控制权
		
		Disconnect(streamStore);
		
		CLog("close");
		call();
	}
	
	
	
	
	
	/*模拟一段录音数据，后面可以调用stop进行编码，需提供pcm数据[1,2,3...]，pcm的采样率*/
	,mock:function(pcmData,pcmSampleRate){
		var This=this;
		This._stop();//清理掉已有的资源
		
		This.isMock=1;
		This.mockEnvInfo=null;
		This.buffers=[pcmData];
		This.recSize=pcmData.length;
		This.srcSampleRate=pcmSampleRate;
		return This;
	}
	,envCheck:function(envInfo){//平台环境下的可用性检查，任何时候都可以调用检查，返回errMsg:""正常，"失败原因"
		//envInfo={envName:"H5",canProcess:true}
		var errMsg,This=this,set=This.set;
		
		//编码器检查环境下配置是否可用
		if(!errMsg){
			if(This[set.type+"_envCheck"]){//编码器已实现环境检查
				errMsg=This[set.type+"_envCheck"](envInfo,set);
			}else{//未实现检查的手动检查配置是否有效
				if(set.takeoffEncodeChunk){
					errMsg=set.type+"类型不支持设置takeoffEncodeChunk";
				};
			};
		};
		
		return errMsg||"";
	}
	,envStart:function(mockEnvInfo,sampleRate){//平台环境相关的start调用
		var This=this,set=This.set;
		This.isMock=mockEnvInfo?1:0;//非H5环境需要启用mock，并提供envCheck需要的环境信息
		This.mockEnvInfo=mockEnvInfo;
		This.buffers=[];//数据缓冲
		This.recSize=0;//数据大小
		
		This.envInLast=0;//envIn接收到最后录音内容的时间
		This.envInFirst=0;//envIn接收到的首个录音内容的录制时间
		This.envInFix=0;//补偿的总时间
		This.envInFixTs=[];//补偿计数列表
		
		set.sampleRate=Math.min(sampleRate,set.sampleRate);//engineCtx需要提前确定最终的采样率
		This.srcSampleRate=sampleRate;
		
		This.engineCtx=0;
		//此类型有边录边转码(Worker)支持
		if(This[set.type+"_start"]){
			var engineCtx=This.engineCtx=This[set.type+"_start"](set);
			if(engineCtx){
				engineCtx.pcmDatas=[];
				engineCtx.pcmSize=0;
			};
		};
	}
	,envResume:function(){//和平台环境无关的恢复录音
		//重新开始计数
		this.envInFixTs=[];
	}
	,envIn:function(pcm,sum){//和平台环境无关的pcm[Int16]输入
		var This=this,set=This.set,engineCtx=This.engineCtx;
		var bufferSampleRate=This.srcSampleRate;
		var size=pcm.length;
		var powerLevel=Recorder.PowerLevel(sum,size);
		
		var buffers=This.buffers;
		var bufferFirstIdx=buffers.length;//之前的buffer都是经过onProcess处理好的，不允许再修改
		buffers.push(pcm);
		
		//有engineCtx时会被覆盖，这里保存一份
		var buffersThis=buffers;
		var bufferFirstIdxThis=bufferFirstIdx;
		
		//卡顿丢失补偿：因为设备很卡的时候导致H5接收到的数据量不够造成播放时候变速，结果比实际的时长要短，此处保证了不会变短，但不能修复丢失的音频数据造成音质变差。当前算法采用输入时间侦测下一帧是否需要添加补偿帧，需要(6次输入||超过1秒)以上才会开始侦测，如果滑动窗口内丢失超过1/3就会进行补偿
		var now=Date.now();
		var pcmTime=Math.round(size/bufferSampleRate*1000);
		This.envInLast=now;
		if(This.buffers.length==1){//记下首个录音数据的录制时间
			This.envInFirst=now-pcmTime;
		};
		var envInFixTs=This.envInFixTs;
		envInFixTs.splice(0,0,{t:now,d:pcmTime});
		//保留3秒的计数滑动窗口，另外超过3秒的停顿不补偿
		var tsInStart=now,tsPcm=0;
		for(var i=0;i<envInFixTs.length;i++){
			var o=envInFixTs[i];
			if(now-o.t>3000){
				envInFixTs.length=i;
				break;
			};
			tsInStart=o.t;
			tsPcm+=o.d;
		};
		//达到需要的数据量，开始侦测是否需要补偿
		var tsInPrev=envInFixTs[1];
		var tsIn=now-tsInStart;
		var lost=tsIn-tsPcm;
		if( lost>tsIn/3 && (tsInPrev&&tsIn>1000 || envInFixTs.length>=6) ){
			//丢失过多，开始执行补偿
			var addTime=now-tsInPrev.t-pcmTime;//距离上次输入丢失这么多ms
			if(addTime>pcmTime/5){//丢失超过本帧的1/5
				var fixOpen=!set.disableEnvInFix;
				CLog("["+now+"]"+(fixOpen?"":"未")+"补偿"+addTime+"ms",3);
				This.envInFix+=addTime;
				
				//用静默进行补偿
				if(fixOpen){
					var addPcm=new Int16Array(addTime*bufferSampleRate/1000);
					size+=addPcm.length;
					buffers.push(addPcm);
				};
			};
		};
		
		
		var sizeOld=This.recSize,addSize=size;
		var bufferSize=sizeOld+addSize;
		This.recSize=bufferSize;//此值在onProcess后需要修正，可能新数据被修改
		
		
		//此类型有边录边转码(Worker)支持，开启实时转码
		if(engineCtx){
			//转换成set的采样率
			var chunkInfo=Recorder.SampleData(buffers,bufferSampleRate,set.sampleRate,engineCtx.chunkInfo);
			engineCtx.chunkInfo=chunkInfo;
			
			sizeOld=engineCtx.pcmSize;
			addSize=chunkInfo.data.length;
			bufferSize=sizeOld+addSize;
			engineCtx.pcmSize=bufferSize;//此值在onProcess后需要修正，可能新数据被修改
			
			buffers=engineCtx.pcmDatas;
			bufferFirstIdx=buffers.length;
			buffers.push(chunkInfo.data);
			bufferSampleRate=chunkInfo.sampleRate;
		};
		
		var duration=Math.round(bufferSize/bufferSampleRate*1000);
		var bufferNextIdx=buffers.length;
		var bufferNextIdxThis=buffersThis.length;
		
		//允许异步处理buffer数据
		var asyncEnd=function(){
			//重新计算size，异步的早已减去添加的，同步的需去掉本次添加的然后重新计算
			var num=asyncBegin?0:-addSize;
			var hasClear=buffers[0]==null;
			for(var i=bufferFirstIdx;i<bufferNextIdx;i++){
				var buffer=buffers[i];
				if(buffer==null){//已被主动释放内存，比如长时间实时传输录音时
					hasClear=1;
				}else{
					num+=buffer.length;
					
					//推入后台边录边转码
					if(engineCtx&&buffer.length){
						This[set.type+"_encode"](engineCtx,buffer);
					};
				};
			};
			
			//同步清理This.buffers，不管buffers到底清了多少个，buffersThis是使用不到的进行全清
			if(hasClear && engineCtx){
				var i=bufferFirstIdxThis;
				if(buffersThis[0]){
					i=0;
				};
				for(;i<bufferNextIdxThis;i++){
					buffersThis[i]=null;
				};
			};
			
			//统计修改后的size，如果异步发生clear要原样加回来，同步的无需操作
			if(hasClear){
				num=asyncBegin?addSize:0;
				
				buffers[0]=null;//彻底被清理
			};
			if(engineCtx){
				engineCtx.pcmSize+=num;
			}else{
				This.recSize+=num;
			};
		};
		//实时回调处理数据，允许修改或替换上次回调以来新增的数据 ，但是不允许修改已处理过的，不允许增删第一维数组 ，允许将第二维数组任意修改替换成空数组也可以
		var asyncBegin=set.onProcess(buffers,powerLevel,duration,bufferSampleRate,bufferFirstIdx,asyncEnd);
		
		if(asyncBegin===true){
			//开启了异步模式，onProcess已接管buffers新数据，立即清空，避免出现未处理的数据
			var hasClear=0;
			for(var i=bufferFirstIdx;i<bufferNextIdx;i++){
				if(buffers[i]==null){//已被主动释放内存，比如长时间实时传输录音时 ，但又要开启异步模式，此种情况是非法的
					hasClear=1;
				}else{
					buffers[i]=new Int16Array(0);
				};
			};
			
			if(hasClear){
				CLog("未进入异步前不能清除buffers",3);
			}else{
				//还原size，异步结束后再统计仅修改后的size，如果发生clear要原样加回来
				if(engineCtx){
					engineCtx.pcmSize-=addSize;
				}else{
					This.recSize-=addSize;
				};
			};
		}else{
			asyncEnd();
		};
	}
	
	
	
	
	//开始录音，需先调用open；只要open成功时，调用此方法是安全的，如果未open强行调用导致的内部错误将不会有任何提示，stop时自然能得到错误
	,start:function(){
		var This=this,ctx=Recorder.Ctx;
		
		var isOpen=1;
		if(This.set.sourceStream){//直接提供了流，仅判断是否调用了open
			if(!This.Stream){
				isOpen=0;
			}
		}else if(!Recorder.IsOpen()){//监测全局麦克风是否打开并且有效
			isOpen=0;
		};
		if(!isOpen){
			CLog("未open",1);
			return;
		};
		CLog("开始录音");
		
		This._stop();
		This.state=0;
		This.envStart(null,ctx.sampleRate);
		
		//检查open过程中stop是否已经调用过
		if(This._SO&&This._SO+1!=This._S){//上面调用过一次 _stop
			//open未完成就调用了stop，此种情况终止start。也应尽量避免出现此情况
			CLog("start被中断",3);
			return;
		};
		This._SO=0;
		
		var end=function(){
			This.state=1;
			This.resume();
		};
		if(ctx.state=="suspended"){
			ctx.resume().then(function(){
				CLog("ctx resume");
				end();
			});
		}else{
			end();
		};
	}
	/*暂停录音*/
	,pause:function(){
		var This=this;
		if(This.state){
			This.state=2;
			CLog("pause");
			delete This._streamStore().Stream._call[This.id];
		};
	}
	/*恢复录音*/
	,resume:function(){
		var This=this;
		if(This.state){
			This.state=1;
			CLog("resume");
			This.envResume();
			
			This._streamStore().Stream._call[This.id]=function(pcm,sum){
				if(This.state==1){
					This.envIn(pcm,sum);
				};
			};
		};
	}
	
	
	
	
	,_stop:function(keepEngine){
		var This=this,set=This.set;
		if(!This.isMock){
			This._S++;
		};
		if(This.state){
			This.pause();
			This.state=0;
		};
		if(!keepEngine && This[set.type+"_stop"]){
			This[set.type+"_stop"](This.engineCtx);
			This.engineCtx=0;
		};
	}
	/*
	结束录音并返回录音数据blob对象
		True(blob,duration) blob：录音数据audio/mp3|wav格式
							duration：录音时长，单位毫秒
		False(msg)
		autoClose:false 可选，是否自动调用close，默认为false
	*/
	,stop:function(True,False,autoClose){
		var This=this,set=This.set,t1;
		CLog("Stop "+(This.envInLast?This.envInLast-This.envInFirst+"ms 补"+This.envInFix+"ms":"-"));
		
		var end=function(){
			This._stop();//彻底关掉engineCtx
			if(autoClose){
				This.close();
			};
		};
		var err=function(msg){
			CLog("结束录音失败："+msg,1);
			False&&False(msg);
			end();
		};
		var ok=function(blob,duration){
			CLog("结束录音 编码"+(Date.now()-t1)+"ms 音频"+duration+"ms/"+blob.size+"b");
			if(set.takeoffEncodeChunk){//接管了输出，此时blob长度为0
				CLog("启用takeoffEncodeChunk后stop返回的blob长度为0不提供音频数据",3);
			}else if(blob.size<Math.max(100,duration/2)){//1秒小于0.5k？
				err("生成的"+set.type+"无效");
				return;
			};
			True&&True(blob,duration);
			end();
		};
		if(!This.isMock){
			if(!This.state){
				err("未开始录音");
				return;
			};
			This._stop(true);
		};
		var size=This.recSize;
		if(!size){
			err("未采集到录音");
			return;
		};
		if(!This.buffers[0]){
			err("音频被释放");
			return;
		};
		if(!This[set.type]){
			err("未加载"+set.type+"编码器");
			return;
		};
		
		//环境配置检查，此处仅针对mock调用，因为open已经检查过了
		if(This.isMock){
			var checkMsg=This.envCheck(This.mockEnvInfo||{envName:"mock",canProcess:false});//没有提供环境信息的mock时没有onProcess回调
			if(checkMsg){
				err("录音错误："+checkMsg);
				return;
			};
		};
		
		//此类型有边录边转码(Worker)支持
		var engineCtx=This.engineCtx;
		if(This[set.type+"_complete"]&&engineCtx){
			var duration=Math.round(engineCtx.pcmSize/set.sampleRate*1000);//采用后的数据长度和buffers的长度可能微小的不一致，是采样率连续转换的精度问题
			
			t1=Date.now();
			This[set.type+"_complete"](engineCtx,function(blob){
				ok(blob,duration);
			},err);
			return;
		};
		
		//标准UI线程转码，调整采样率
		t1=Date.now();
		var chunk=Recorder.SampleData(This.buffers,This.srcSampleRate,set.sampleRate);
		
		set.sampleRate=chunk.sampleRate;
		var res=chunk.data;
		var duration=Math.round(res.length/set.sampleRate*1000);
		
		CLog("采样"+size+"->"+res.length+" 花:"+(Date.now()-t1)+"ms");
		
		setTimeout(function(){
			t1=Date.now();
			This[set.type](res,function(blob){
				ok(blob,duration);
			},function(msg){
				err(msg);
			});
		});
	}

};

if(window.Recorder){
	window.Recorder.Destroy();
};
window.Recorder=Recorder;

//end ****copy源码结束*****
Recorder.LM=LM;

//流量统计用1像素图片地址，设置为空将不参与统计
Recorder.TrafficImgUrl="//ia.51.la/go1?id=20469973&pvFlag=1";
Recorder.Traffic=function(){
	var imgUrl=Recorder.TrafficImgUrl;
	if(imgUrl){
		var data=Recorder.Traffic;
		var idf=location.href.replace(/#.*/,"");
		
		if(imgUrl.indexOf("//")==0){
			//给url加上http前缀，如果是file协议下，不加前缀没法用
			if(/^https:/i.test(idf)){
				imgUrl="https:"+imgUrl;
			}else{
				imgUrl="http:"+imgUrl;
			};
		};
		
		if(!data[idf]){
			data[idf]=1;
			
			var img=new Image();
			img.src=imgUrl;
			CLog("Traffic Analysis Image: Recorder.TrafficImgUrl="+Recorder.TrafficImgUrl);
		};
	};
};

}));