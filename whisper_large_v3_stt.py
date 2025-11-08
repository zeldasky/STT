"""
Whisper Large-v3 기반 한국어 음성-텍스트 변환 시스템
최고 정확도, 구두점 자동 처리, 자연스러운 띄어쓰기
"""

import whisper
import warnings
import os
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import torch

warnings.filterwarnings('ignore')


class WhisperLargeV3STT:
    """Whisper Large-v3 기반 한국어 STT 시스템"""

    def __init__(self, use_gpu=False):
        """
        Args:
            use_gpu: GPU 사용 여부 (False=CPU, True=CUDA)
        """
        print("=" * 80)
        print("🎙️ Whisper Large-v3 한국어 STT 시스템")
        print("=" * 80)
        print()

        # 디바이스 설정
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            print("🚀 NVIDIA GPU 사용")
        else:
            self.device = "cpu"
            print("💻 CPU 사용 (안정성 우선)")

        print()
        print("🔄 Whisper Large-v3 모델 로딩 중...")
        print("   (처음 실행 시 약 3GB 다운로드 - 시간이 걸릴 수 있습니다)")
        print()

        try:
            # Whisper large-v3 모델 로드
            self.model = whisper.load_model("large-v3", device=self.device)
            print("✅ Whisper Large-v3 모델 로딩 완료!")
            print()
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            print()
            print("💡 해결 방법:")
            print("   pip install --upgrade openai-whisper")
            raise

    def transcribe_file(self, audio_path, language="ko", verbose=True):
        """
        오디오 파일을 텍스트로 변환

        Args:
            audio_path: 오디오 파일 경로
            language: 언어 코드 (기본값: "ko" - 한국어)
            verbose: 진행 상황 표시 여부

        Returns:
            dict: 변환 결과 (segments, text 포함)
        """
        print("=" * 80)
        print(f"📂 오디오 파일: {audio_path}")
        print("=" * 80)
        print()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_path}")

        # 파일 정보
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        print(f"📊 파일 크기: {file_size:.2f} MB")
        print()

        print("🚀 음성 인식 시작...")
        print("   (긴 오디오는 시간이 걸릴 수 있습니다)")
        print()

        try:
            # Whisper 변환 수행
            result = self.model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                verbose=verbose,
                fp16=False,  # CPU 호환성
                temperature=0.0,  # 일관성 있는 결과
                beam_size=5,  # 정확도 향상
                best_of=5,  # 최고 품질
                patience=1.0  # 안정성
            )

            print()
            print("✅ 음성 인식 완료!")
            print()

            return result

        except Exception as e:
            print(f"❌ 변환 중 오류 발생: {e}")
            raise

    def format_results(self, result):
        """
        Whisper 결과를 표준 형식으로 변환

        Args:
            result: Whisper 변환 결과

        Returns:
            list: 표준화된 세그먼트 리스트
        """
        formatted_results = []

        for segment in result['segments']:
            formatted_results.append({
                'id': segment['id'],
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'speaker': 'Speaker_0'  # 기본 화자
            })

        return formatted_results

    def save_results(self, result, formatted_results, output_dir="output", audio_filename="audio"):
        """
        결과를 다양한 형식으로 저장

        Args:
            result: 원본 Whisper 결과
            formatted_results: 표준화된 결과
            output_dir: 출력 디렉토리
            audio_filename: 원본 오디오 파일명
        """
        Path(output_dir).mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{audio_filename}_{timestamp}"

        print("=" * 80)
        print("💾 결과 저장 중...")
        print("=" * 80)
        print()

        # 1. 전체 텍스트 저장 (구두점 포함)
        full_text_path = f"{output_dir}/{base_name}_full.txt"
        with open(full_text_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        print(f"✅ 전체 텍스트: {full_text_path}")

        # 2. 타임스탬프 포함 텍스트
        timestamped_path = f"{output_dir}/{base_name}_timestamped.txt"
        with open(timestamped_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("회의록 자동 변환 결과 (Whisper Large-v3)\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for item in formatted_results:
                start_min = int(item['start'] // 60)
                start_sec = int(item['start'] % 60)
                end_min = int(item['end'] // 60)
                end_sec = int(item['end'] % 60)

                f.write(f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]\n")
                f.write(f"{item['text']}\n\n")
        print(f"✅ 타임스탬프 텍스트: {timestamped_path}")

        # 3. JSON 저장 (프로그래밍 용도)
        json_path = f"{output_dir}/{base_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'model': 'whisper-large-v3',
                    'language': result.get('language', 'ko'),
                    'duration': formatted_results[-1]['end'] if formatted_results else 0,
                    'timestamp': datetime.now().isoformat()
                },
                'full_text': result['text'],
                'segments': formatted_results
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON 파일: {json_path}")

        # 4. 마크다운 저장 (읽기 좋은 형식)
        md_path = f"{output_dir}/{base_name}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# 회의록\n\n")
            f.write(f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**모델**: Whisper Large-v3\n\n")
            f.write(f"**총 길이**: {formatted_results[-1]['end'] / 60:.1f}분\n\n" if formatted_results else "")
            f.write("---\n\n")

            f.write("## 전체 내용\n\n")
            f.write(result['text'] + "\n\n")

            f.write("---\n\n")
            f.write("## 타임스탬프별 내용\n\n")

            for item in formatted_results:
                start_min = int(item['start'] // 60)
                start_sec = int(item['start'] % 60)
                end_min = int(item['end'] // 60)
                end_sec = int(item['end'] % 60)

                f.write(f"### [{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]\n\n")
                f.write(f"{item['text']}\n\n")
        print(f"✅ 마크다운 파일: {md_path}")

        # 5. SRT 자막 파일 (영상 자막용)
        srt_path = f"{output_dir}/{base_name}.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, item in enumerate(formatted_results, 1):
                start_time = self._format_srt_time(item['start'])
                end_time = self._format_srt_time(item['end'])

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{item['text']}\n\n")
        print(f"✅ SRT 자막 파일: {srt_path}")

        print()
        print("=" * 80)
        print("✅ 모든 파일 저장 완료!")
        print("=" * 80)

    def _format_srt_time(self, seconds):
        """초를 SRT 시간 형식으로 변환 (00:00:00,000)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def print_summary(self, result, formatted_results):
        """결과 요약 출력"""
        print()
        print("=" * 80)
        print("📊 변환 결과 요약")
        print("=" * 80)
        print()
        print(f"🎯 총 세그먼트: {len(formatted_results)}개")
        print(f"⏱️  총 길이: {formatted_results[-1]['end'] / 60:.1f}분" if formatted_results else "")
        print(f"📝 총 글자 수: {len(result['text'])}자")
        print(f"🗣️  언어: {result.get('language', 'ko').upper()}")
        print()

        # 처음 3개 세그먼트 미리보기
        print("=" * 80)
        print("📄 변환 결과 미리보기 (처음 3개)")
        print("=" * 80)
        print()

        for item in formatted_results[:3]:
            start_min = int(item['start'] // 60)
            start_sec = int(item['start'] % 60)
            end_min = int(item['end'] // 60)
            end_sec = int(item['end'] % 60)

            print(f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]")
            print(f"{item['text']}")
            print()

        if len(formatted_results) > 3:
            print(f"... 외 {len(formatted_results) - 3}개 세그먼트")
            print()


def main():
    """메인 실행 함수"""

    print()
    print("=" * 80)
    print("🎙️ Whisper Large-v3 한국어 음성-텍스트 변환 시스템")
    print("=" * 80)
    print()
    print("✨ 특징:")
    print("   - 최고 수준의 한국어 인식 정확도")
    print("   - 자동 구두점 처리")
    print("   - 자연스러운 띄어쓰기")
    print("   - 타임스탬프 자동 생성")
    print()
    print("=" * 80)
    print()

    # GPU 사용 여부 선택
    use_gpu_input = input("🚀 GPU를 사용하시겠습니까? (y/n, 기본값=n): ").strip().lower()
    use_gpu = use_gpu_input == 'y'

    print()

    # STT 시스템 초기화
    try:
        stt_system = WhisperLargeV3STT(use_gpu=use_gpu)
    except Exception as e:
        print(f"\n❌ 시스템 초기화 실패")
        return

    # 오디오 파일 경로 입력
    print("=" * 80)
    audio_path = input("🎵 오디오 파일 경로를 입력하세요 (드래그 앤 드롭 가능): ").strip()
    audio_path = audio_path.replace('\\ ', ' ').strip("'\"")

    if not os.path.exists(audio_path):
        print(f"❌ 파일을 찾을 수 없습니다: {audio_path}")
        return

    print()

    # 출력 디렉토리 설정
    output_dir = input("📁 출력 디렉토리 (기본값=output): ").strip() or "output"

    print()
    print("=" * 80)
    print("⚙️ 설정 확인")
    print("=" * 80)
    print(f"📂 입력 파일: {audio_path}")
    print(f"📁 출력 디렉토리: {output_dir}")
    print(f"🖥️  디바이스: {'GPU (CUDA)' if use_gpu else 'CPU'}")
    print(f"🤖 모델: Whisper Large-v3")
    print("=" * 80)
    print()

    confirm = input("계속 진행하시겠습니까? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 취소되었습니다.")
        return

    print()

    try:
        # STT 수행
        result = stt_system.transcribe_file(audio_path, language="ko", verbose=True)

        # 결과 포맷팅
        formatted_results = stt_system.format_results(result)

        # 결과 저장
        audio_filename = Path(audio_path).stem
        stt_system.save_results(result, formatted_results, output_dir, audio_filename)

        # 요약 출력
        stt_system.print_summary(result, formatted_results)

        print()
        print("=" * 80)
        print("✅ 모든 작업이 완료되었습니다!")
        print("=" * 80)
        print()
        print(f"📁 결과 파일 위치: {output_dir}/")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()