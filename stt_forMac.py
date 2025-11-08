"""
Mac용 한국어 음성-텍스트 변환 시스템 (MPS 문제 해결 버전)
"""

import torch
import torchaudio
import librosa
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyannote.audio import Pipeline
import warnings
import os
from pathlib import Path
from datetime import datetime
import json
import numpy as np

warnings.filterwarnings('ignore')


class KoreanSTTSystem:
    """한국어 음성 인식 시스템"""

    def __init__(self, model_name="kresnik/wav2vec2-large-xlsr-korean", huggingface_token=None, use_gpu=False):
        """
        Args:
            model_name: HuggingFace 모델명
            huggingface_token: HuggingFace API 토큰
            use_gpu: GPU 사용 여부 (기본값: False - CPU 사용)
        """
        print("🔄 모델 로딩 중...")

        # ⚠️ MPS 문제 해결: CPU 강제 사용
        if use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
                print("🚀 NVIDIA GPU 사용")
            else:
                self.device = "cpu"
                print("💻 CPU 사용 (CUDA 미지원)")
        else:
            self.device = "cpu"
            print("💻 CPU 사용 (안정성 우선)")

        # STT 모델 로드
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(
                model_name,
                token=huggingface_token
            )
            self.model = Wav2Vec2ForCTC.from_pretrained(
                model_name,
                token=huggingface_token
            ).to(self.device)
            print(f"✅ STT 모델 로딩 완료: {model_name}")
        except Exception as e:
            print(f"❌ STT 모델 로딩 실패: {e}")
            print("💡 해결방법: huggingface-cli login 실행 후 토큰 입력")
            raise

        # 화자 분리 모델 (선택사항)
        self.diarization_pipeline = None
        if huggingface_token:
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=huggingface_token
                )
                print("✅ 화자 분리 모델 로딩 완료")
            except Exception as e:
                print(f"⚠️ 화자 분리 모델 로딩 실패: {e}")
                print("💡 화자 분리 없이 STT만 진행합니다.")

    def load_audio(self, audio_path, target_sr=16000):
        """오디오 파일 로드 및 전처리"""
        print(f"📂 오디오 로딩: {audio_path}")

        # librosa로 오디오 로드
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

        print(f"✅ 오디오 로드 완료 (길이: {len(audio) / sr:.2f}초, SR: {sr}Hz)")
        return audio, sr

    def transcribe_audio(self, audio, sr=16000, chunk_length_s=30):
        """
        오디오를 텍스트로 변환 (긴 오디오는 청크로 분할)

        Args:
            audio: 오디오 데이터 (numpy array)
            sr: 샘플링 레이트
            chunk_length_s: 청크 길이 (초)

        Returns:
            str: 변환된 텍스트
        """
        audio_length = len(audio) / sr

        # 짧은 오디오는 한 번에 처리
        if audio_length <= chunk_length_s:
            return self._transcribe_chunk(audio, sr)

        # 긴 오디오는 청크로 분할 처리
        print(f"📊 긴 오디오 감지 ({audio_length:.1f}초) - 청크 단위로 처리합니다...")

        chunk_samples = int(chunk_length_s * sr)
        transcriptions = []

        num_chunks = int(np.ceil(len(audio) / chunk_samples))

        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = min((i + 1) * chunk_samples, len(audio))

            chunk = audio[start_idx:end_idx]

            print(f"  처리 중: {i + 1}/{num_chunks} ({start_idx / sr:.1f}s - {end_idx / sr:.1f}s)")

            text = self._transcribe_chunk(chunk, sr)
            transcriptions.append(text)

        return " ".join(transcriptions)

    def _transcribe_chunk(self, audio, sr=16000):
        """단일 청크 변환"""
        # 오디오 전처리
        input_values = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        ).input_values.to(self.device)

        # 추론
        with torch.no_grad():
            logits = self.model(input_values).logits

        # 디코딩
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription

    def diarize_speakers(self, audio_path):
        """화자 분리 수행"""
        if self.diarization_pipeline is None:
            print("⚠️ 화자 분리 모델이 로드되지 않았습니다.")
            return None

        print("🎤 화자 분리 진행 중...")

        try:
            diarization = self.diarization_pipeline(audio_path)

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })

            print(f"✅ 화자 분리 완료: {len(set([s['speaker'] for s in segments]))}명 감지")
            return segments

        except Exception as e:
            print(f"❌ 화자 분리 실패: {e}")
            return None

    def transcribe_with_speakers(self, audio_path):
        """화자 분리 + STT 통합 수행"""
        # 오디오 로드
        audio, sr = self.load_audio(audio_path)

        # 화자 분리
        speaker_segments = self.diarize_speakers(audio_path)

        results = []

        if speaker_segments:
            # 화자별로 STT 수행
            print("📝 화자별 텍스트 변환 중...")

            for i, segment in enumerate(speaker_segments):
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)

                segment_audio = audio[start_sample:end_sample]

                if len(segment_audio) > sr * 0.5:  # 0.5초 이상만 처리
                    text = self.transcribe_audio(segment_audio, sr)

                    results.append({
                        'speaker': segment['speaker'],
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': text.strip()
                    })

                    print(f"  [{segment['speaker']}] {segment['start']:.1f}s-{segment['end']:.1f}s: {text[:50]}...")

        else:
            # 화자 분리 없이 전체 STT
            print("📝 전체 텍스트 변환 중...")
            text = self.transcribe_audio(audio, sr)

            results.append({
                'speaker': 'Speaker_0',
                'start': 0.0,
                'end': len(audio) / sr,
                'text': text.strip()
            })

        print(f"✅ 변환 완료: {len(results)}개 세그먼트")
        return results

    def save_results(self, results, output_dir="output"):
        """결과를 파일로 저장"""
        Path(output_dir).mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. JSON 저장
        json_path = f"{output_dir}/transcript_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON 저장: {json_path}")

        # 2. 텍스트 저장
        txt_path = f"{output_dir}/transcript_{timestamp}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("회의록 자동 변환 결과\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            current_speaker = None
            for item in results:
                if item['speaker'] != current_speaker:
                    f.write(f"\n[{item['speaker']}]\n")
                    current_speaker = item['speaker']

                f.write(f"[{item['start']:.1f}s - {item['end']:.1f}s]\n")
                f.write(f"{item['text']}\n\n")

        print(f"💾 텍스트 저장: {txt_path}")

        # 3. 마크다운 저장
        md_path = f"{output_dir}/transcript_{timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# 회의록\n\n")
            f.write(f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**참석자**: {', '.join(set([r['speaker'] for r in results]))}\n\n")
            f.write("---\n\n")

            current_speaker = None
            for item in results:
                if item['speaker'] != current_speaker:
                    f.write(f"\n## {item['speaker']}\n\n")
                    current_speaker = item['speaker']

                f.write(f"**[{item['start']:.1f}s - {item['end']:.1f}s]**\n\n")
                f.write(f"{item['text']}\n\n")

        print(f"💾 마크다운 저장: {md_path}")


def main():
    """메인 실행 함수"""

    print("=" * 80)
    print("🎙️ Mac용 한국어 음성-텍스트 변환 시스템 (MPS 문제 해결 버전)")
    print("=" * 80)
    print()

    # HuggingFace 토큰
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    if not hf_token:
        print("⚠️ HuggingFace 토큰이 설정되지 않았습니다.")
        print("💡 화자 분리 기능을 사용하려면 토큰이 필요합니다.")
        print("💡 토큰 없이는 STT만 가능합니다.")
        print()
        use_token = input("토큰을 입력하시겠습니까? (y/n): ").lower()
        if use_token == 'y':
            hf_token = input("HuggingFace 토큰 입력: ").strip()

    # STT 시스템 초기화 (use_gpu=False로 CPU 강제 사용)
    try:
        stt_system = KoreanSTTSystem(
            model_name="kresnik/wav2vec2-large-xlsr-korean",
            huggingface_token=hf_token,
            use_gpu=False  # ⚠️ CPU 강제 사용
        )
    except Exception as e:
        print(f"\n❌ 시스템 초기화 실패: {e}")
        print("\n💡 해결 방법:")
        print("1. 터미널에서 'huggingface-cli login' 실행")
        print("2. https://huggingface.co/settings/tokens 에서 토큰 발급")
        print("3. 토큰 입력 후 다시 시도")
        return

    print()
    print("=" * 80)

    # 오디오 파일 경로 입력
    audio_path = input("🎵 오디오 파일 경로를 입력하세요 (드래그 앤 드롭 가능): ").strip()
    audio_path = audio_path.replace('\\ ', ' ').strip("'\"")

    if not os.path.exists(audio_path):
        print(f"❌ 파일을 찾을 수 없습니다: {audio_path}")
        return

    print()
    print("=" * 80)
    print("🚀 변환 시작...")
    print("=" * 80)
    print()

    try:
        # STT 수행
        results = stt_system.transcribe_with_speakers(audio_path)

        # 결과 저장
        stt_system.save_results(results)

        print()
        print("=" * 80)
        print("✅ 모든 작업 완료!")
        print("=" * 80)
        print()
        print("📊 결과 요약:")
        print(f"  - 총 세그먼트: {len(results)}개")
        print(f"  - 화자 수: {len(set([r['speaker'] for r in results]))}명")
        print(f"  - 총 텍스트 길이: {sum([len(r['text']) for r in results])}자")
        print()
        print("📁 저장된 파일: output/ 디렉토리 확인")

    except Exception as e:
        print(f"\n❌ 변환 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
