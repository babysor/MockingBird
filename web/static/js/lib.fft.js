/*
时域转频域，快速傅里叶变换(FFT)
https://github.com/xiangyuecn/Recorder

var fft=Recorder.LibFFT(bufferSize)
	bufferSize取值2的n次方

fft.bufferSize 实际采用的bufferSize
fft.transform(inBuffer)
	inBuffer:[Int16,...] 数组长度必须是bufferSize
	返回[Float64(Long),...]，长度为bufferSize/2
*/

/*
从FFT.java 移植，Java开源库：jmp123 版本0.3
https://www.iteye.com/topic/851459
https://sourceforge.net/projects/jmp123/files/
*/
Recorder.LibFFT=function(bufferSize){
	"use strict";
	
	var FFT_N_LOG,FFT_N,MINY;
	var real, imag, sintable, costable;
	var bitReverse;

	var FFT_Fn=function(bufferSize) {//bufferSize只能取值2的n次方
		FFT_N_LOG=Math.round(Math.log(bufferSize)/Math.log(2));
		FFT_N = 1 << FFT_N_LOG;
		MINY = ((FFT_N << 2) * Math.sqrt(2));
		
		real = [];
		imag = [];
		sintable = [0];
		costable = [0];
		bitReverse = [];

		var i, j, k, reve;
		for (i = 0; i < FFT_N; i++) {
			k = i;
			for (j = 0, reve = 0; j != FFT_N_LOG; j++) {
				reve <<= 1;
				reve |= (k & 1);
				k >>>= 1;
			}
			bitReverse[i] = reve;
		}

		var theta, dt = 2 * Math.PI / FFT_N;
		for (i = (FFT_N >> 1) - 1; i > 0; i--) {
			theta = i * dt;
			costable[i] = Math.cos(theta);
			sintable[i] = Math.sin(theta);
		}
	}

	/*
	用于频谱显示的快速傅里叶变换 
    inBuffer 输入FFT_N个实数，返回 FFT_N/2个输出值(复数模的平方)。 
	*/
	var getModulus=function(inBuffer) {
		var i, j, k, ir, j0 = 1, idx = FFT_N_LOG - 1;
		var cosv, sinv, tmpr, tmpi;
		for (i = 0; i != FFT_N; i++) {
			real[i] = inBuffer[bitReverse[i]];
			imag[i] = 0;
		}

		for (i = FFT_N_LOG; i != 0; i--) {
			for (j = 0; j != j0; j++) {
				cosv = costable[j << idx];
				sinv = sintable[j << idx];
				for (k = j; k < FFT_N; k += j0 << 1) {
					ir = k + j0;
					tmpr = cosv * real[ir] - sinv * imag[ir];
					tmpi = cosv * imag[ir] + sinv * real[ir];
					real[ir] = real[k] - tmpr;
					imag[ir] = imag[k] - tmpi;
					real[k] += tmpr;
					imag[k] += tmpi;
				}
			}
			j0 <<= 1;
			idx--;
		}

		j = FFT_N >> 1;
		var outBuffer=new Float64Array(j);
		/*
		 * 输出模的平方:
		 * for(i = 1; i <= j; i++)
		 * 	inBuffer[i-1] = real[i] * real[i] +  imag[i] * imag[i];
		 * 
		 * 如果FFT只用于频谱显示,可以"淘汰"幅值较小的而减少浮点乘法运算. MINY的值
		 * 和Spectrum.Y0,Spectrum.logY0对应.
		 */
		sinv = MINY;
		cosv = -MINY;
		for (i = j; i != 0; i--) {
			tmpr = real[i];
			tmpi = imag[i];
			if (tmpr > cosv && tmpr < sinv && tmpi > cosv && tmpi < sinv)
				outBuffer[i - 1] = 0;
			else
				outBuffer[i - 1] = Math.round(tmpr * tmpr + tmpi * tmpi);
		}
		return outBuffer;
	}
	
	FFT_Fn(bufferSize);
	return {transform:getModulus,bufferSize:FFT_N};
};
