"""
윈도우 스피커 출력 실시간 한국어 STT 시스템
Whisper 기반 - 스피커에서 나오는 소리를 실시간으로 텍스트 변환
webrtcvad 없이 작동 (numpy만 사용)
"""

import whisper
import numpy as np
import soundcard as sc
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class RealtimeSpeakerSTT:
    """실시간 스피커 출력 음성 인식 시스템"""

    def __init__(self, model_size="base", language="ko"):
        """
        Args:
            model_size: "tiny", "base", "small", "medium", "large-v3"
                       실시간 처리를 위해 "base" 또는 "small" 추천
            language: 언어 코드 (기본값: "ko" - 한국어)
        """
        print("=" * 80)
        print("🎙️ 실시간 스피커 출력 STT 시스템")
        print("=" * 80)
        print()

        self.model_size = model_size
        self.language = language
        self.sample_rate = 32000
        self.chunk_duration = 5.0  # 3초마다 변환
        self.is_running = False

        # 오디오 버퍼
        self.audio_queue = queue.Queue()
        self.text_results = []

        # 음성 감지 설정
        self.silence_threshold = 0.005  # 음량 임계값 (조절 가능)
        self.min_speech_duration = 0.5  # 최소 음성 길이 (초)

        # 모델 로드
        print(f"🔄 Whisper {model_size} 모델 로딩 중...")
        print("   (처음 실행 시 다운로드가 필요할 수 있습니다)")
        self.model = whisper.load_model(model_size)
        print(f"✅ 모델 로딩 완료!")
        print()

    def list_speakers(self):
        """사용 가능한 스피커(출력 장치) 목록 표시"""
        print("=" * 80)
        print("🔊 사용 가능한 스피커 (출력 장치)")
        print("=" * 80)
        print()

        try:
            # 모든 스피커 가져오기
            speakers = sc.all_speakers()

            if not speakers:
                print("❌ 사용 가능한 스피커를 찾을 수 없습니다.")
                print()
                print("💡 해결 방법:")
                print("   1. 윈도우 설정 → 시스템 → 소리 → 출력 장치 확인")
                print("   2. 스테레오 믹스 활성화:")
                print("      제어판 → 소리 → 녹음 탭 → 스테레오 믹스 우클릭 → 활성화")
                return None

            for i, speaker in enumerate(speakers):
                is_default = "(기본값)" if i == 0 else ""
                print(f"{i+1}. {speaker.name} {is_default}")

            print()
            return speakers

        except Exception as e:
            print(f"❌ 스피커 목록 가져오기 실패: {e}")
            print()
            print("💡 관리자 권한으로 실행해보세요.")
            return None

    def select_speaker(self):
        """스피커 선택"""
        speakers = self.list_speakers()

        if not speakers:
            return None

        while True:
            try:
                choice = input(f"스피커 선택 (1-{len(speakers)}, 기본값=1): ").strip()

                if not choice:
                    choice = "1"

                idx = int(choice) - 1

                if 0 <= idx < len(speakers):
                    selected = speakers[idx]
                    print(f"✅ 선택됨: {selected.name}")
                    print()
                    return selected
                else:
                    print(f"❌ 1-{len(speakers)} 사이의 숫자를 입력하세요.")
            except ValueError:
                print("❌ 올바른 숫자를 입력하세요.")
            except KeyboardInterrupt:
                print("\n❌ 취소되었습니다.")
                return None

    def is_speech(self, audio_chunk):
        """
        오디오 청크에 음성이 있는지 간단히 판단
        (webrtcvad 대신 음량 기반 감지)
        """
        # RMS (Root Mean Square) 계산
        rms = np.sqrt(np.mean(audio_chunk**2))

        # 임계값 이상이면 음성으로 판단
        return rms > self.silence_threshold

    def audio_capture_thread(self, speaker):
        """오디오 캡처 스레드"""
        print("🎧 오디오 캡처 시작...")

        chunk_samples = int(self.chunk_duration * self.sample_rate)

        try:
            # 스피커 출력을 루프백으로 녹음
            with sc.get_microphone(
                id=str(speaker.name),
                include_loopback=True
            ).recorder(samplerate=self.sample_rate, channels=1) as mic:

                print("✅ 녹음 준비 완료")
                print()

                while self.is_running:
                    # 오디오 청크 녹음
                    audio_chunk = mic.record(numframes=chunk_samples)

                    # 모노로 변환
                    if len(audio_chunk.shape) > 1:
                        audio_chunk = audio_chunk.mean(axis=1)

                    audio_flat = audio_chunk.flatten()

                    # 음성이 있는지 체크
                    if self.is_speech(audio_flat):
                        # 큐에 추가
                        self.audio_queue.put(audio_flat)

        except Exception as e:
            print(f"❌ 오디오 캡처 오류: {e}")
            print()
            print("💡 해결 방법:")
            print("   1. 다른 스피커를 선택해보세요")
            print("   2. 스테레오 믹스를 활성화하세요")
            print("   3. 프로그램을 관리자 권한으로 실행하세요")
            self.is_running = False

    def transcribe_thread(self):
        """음성 인식 스레드"""
        print("📝 음성 인식 준비 완료")
        print()
        print("=" * 80)
        print("🎬 변환 시작! 스피커에서 소리를 내보세요")
        print("=" * 80)
        print()

        segment_count = 0

        while self.is_running:
            try:
                # 큐에서 오디오 가져오기 (타임아웃 1초)
                audio_chunk = self.audio_queue.get(timeout=1.0)

                # Whisper로 변환
                result = self.model.transcribe(
                    audio_chunk,
                    language=self.language,
                    task="transcribe",
                    fp16=True,
                    verbose=False,
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6
                )

                text = result['text'].strip()

                # 텍스트가 있고 의미있는 길이면 출력
                if text and len(text) > 1:
                    segment_count += 1
                    timestamp = datetime.now().strftime("%H:%M:%S")

                    # 컬러 출력 (윈도우 터미널 지원)
                    print(f"[{timestamp}] #{segment_count:03d}: {text}")

                    # 결과 저장
                    self.text_results.append({
                        'timestamp': timestamp,
                        'segment': segment_count,
                        'text': text
                    })

            except queue.Empty:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"⚠️ 변환 오류: {e}")

    def start(self, speaker):
        """실시간 STT 시작"""
        print("=" * 80)
        print("🚀 실시간 변환 시작")
        print("=" * 80)
        print()
        print("💡 사용 방법:")
        print("   1. 노트북에서 유튜브, 영상, 음악 등을 재생하세요")
        print("   2. 스피커로 나오는 소리가 자동으로 텍스트로 변환됩니다")
        print("   3. 종료하려면 Ctrl+C를 누르세요")
        print()
        print(f"⚙️  설정:")
        print(f"   - 모델: Whisper {self.model_size}")
        print(f"   - 청크 길이: {self.chunk_duration}초")
        print(f"   - 음량 임계값: {self.silence_threshold}")
        print()
        print("=" * 80)
        print()

        self.is_running = True

        # 오디오 캡처 스레드 시작
        capture_thread = threading.Thread(
            target=self.audio_capture_thread,
            args=(speaker,),
            daemon=True
        )
        capture_thread.start()

        # 음성 인식 스레드 시작
        transcribe_thread = threading.Thread(
            target=self.transcribe_thread,
            daemon=True
        )
        transcribe_thread.start()

        try:
            # 메인 스레드는 대기
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n⚠️ 종료 중...")
            self.stop()

    def stop(self):
        """실시간 STT 중지"""
        self.is_running = False
        time.sleep(1)  # 스레드 종료 대기

        print()
        print("=" * 80)
        print("✅ 변환 종료")
        print("=" * 80)
        print()
        print(f"📊 총 {len(self.text_results)}개 세그먼트 변환됨")
        print()

    def save_results(self, output_dir="output"):
        """결과 저장"""
        if not self.text_results:
            print("💾 저장할 결과가 없습니다.")
            return

        Path(output_dir).mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("=" * 80)
        print("💾 결과 저장 중...")
        print("=" * 80)
        print()

        # 1. 타임스탬프 포함 텍스트 파일
        txt_path = f"{output_dir}/realtime_transcript_{timestamp}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("실시간 스피커 출력 변환 결과\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"모델: Whisper {self.model_size}\n")
            f.write(f"총 세그먼트: {len(self.text_results)}개\n")
            f.write("=" * 80 + "\n\n")

            for item in self.text_results:
                f.write(f"[{item['timestamp']}] #{item['segment']:03d}\n")
                f.write(f"{item['text']}\n\n")

        print(f"✅ 타임스탬프 텍스트: {txt_path}")

        # 2. 전체 텍스트만 (구두점 포함)
        full_text_path = f"{output_dir}/realtime_transcript_{timestamp}_full.txt"
        with open(full_text_path, 'w', encoding='utf-8') as f:
            full_text = " ".join([item['text'] for item in self.text_results])
            f.write(full_text)

        print(f"✅ 전체 텍스트: {full_text_path}")

        # 3. 마크다운 형식
        md_path = f"{output_dir}/realtime_transcript_{timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# 실시간 변환 결과\n\n")
            f.write(f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**모델**: Whisper {self.model_size}\n\n")
            f.write(f"**총 세그먼트**: {len(self.text_results)}개\n\n")
            f.write("---\n\n")

            f.write("## 전체 내용\n\n")
            full_text = " ".join([item['text'] for item in self.text_results])
            f.write(full_text + "\n\n")

            f.write("---\n\n")
            f.write("## 타임스탬프별 내용\n\n")

            for item in self.text_results:
                f.write(f"### [{item['timestamp']}] #{item['segment']:03d}\n\n")
                f.write(f"{item['text']}\n\n")

        print(f"✅ 마크다운: {md_path}")

        print()
        print("=" * 80)
        print("✅ 모든 파일 저장 완료!")
        print("=" * 80)
        print()
        print(f"📁 저장 위치: {output_dir}/")
        print()


def main():
    """메인 실행 함수"""

    print()
    print("=" * 80)
    print("🎙️ 실시간 스피커 출력 → 한국어 STT 시스템")
    print("=" * 80)
    print()
    print("✨ 특징:")
    print("   - 윈도우 스피커 출력 실시간 캡처")
    print("   - 유튜브, 영상, 음악 등 모든 소리 변환")
    print("   - 한국어 자동 인식")
    print("   - 실시간 텍스트 출력")
    print("   - 자동 구두점 처리")
    print()
    print("=" * 80)
    print()

    # 모델 크기 선택
    print("📊 모델 크기 선택 (실시간 처리용):")
    print("  1. tiny    - 매우 빠름 (정확도 낮음) ⚡")
    print("  2. base    - 빠름 (정확도 보통) ⭐ 추천!")
    print("  3. small   - 중간 (정확도 좋음)")
    print("  4. medium  - 느림 (정확도 높음)")
    print()

    choice = input("선택 (1-4, 기본값=2): ").strip() or "2"

    model_sizes = {
        "1": "tiny",
        "2": "base",
        "3": "small",
        "4": "medium"
    }

    model_size = model_sizes.get(choice, "base")

    print()
    print(f"✅ {model_size} 모델 선택됨")
    print()

    # STT 시스템 초기화
    try:
        stt_system = RealtimeSpeakerSTT(
            model_size=model_size,
            language="ko"
        )
    except Exception as e:
        print(f"\n❌ 시스템 초기화 실패: {e}")
        print()
        print("💡 해결 방법:")
        print("   pip install --upgrade openai-whisper torch")
        return

    # 스피커 선택
    speaker = stt_system.select_speaker()

    if speaker is None:
        print("❌ 스피커를 선택하지 않았습니다.")
        print()
        print("💡 스테레오 믹스 활성화 방법:")
        print("   1. 윈도우 검색 → '소리 설정'")
        print("   2. 고급 → 녹음 탭")
        print("   3. 스테레오 믹스 우클릭 → 활성화")
        print("   4. 프로그램 재실행")
        return

    # 실시간 STT 시작
    try:
        stt_system.start(speaker)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 결과 저장
        if stt_system.text_results:
            print()
            save_choice = input("💾 결과를 저장하시겠습니까? (y/n, 기본값=y): ").strip().lower()
            if save_choice != 'n':
                stt_system.save_results()

        print()
        print("👋 프로그램을 종료합니다.")
        print()


if __name__ == "__main__":
    main()