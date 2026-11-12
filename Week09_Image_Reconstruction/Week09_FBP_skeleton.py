# %% ============================================================
# BMED318 Medical Image Processing | Week 09: FBP Skeleton
#
# 학습 목표
#   1) Ramp / Butterworth-windowed ramp 주파수 필터 만들기
#   2) 각 1D projection에 주파수 영역 필터링 적용하기
#   3) 각도별 filtered projection을 회전시켜 backprojection 수행하기
#   4) 모든 backprojection 결과를 더하여(superposition) 영상 복원하기
#
# 참고: Image Recon FBP.pdf / Week09.Handson.pdf
# NOTE: scikit-image의 iradon()을 이용하지 않고 직접 구현
# TODO 1~8을 직접 작성하고 해당 raise NotImplementedError 줄은 삭제하세요.
# ================================================================

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage import io


# %% [예제] ndi.rotate() 동작 확인 해당 셀은 실행만 해보시면 됩니다.
def show_rotation_example():
    """reshape=False가 원본 FOV(배열 크기)를 유지하는 방법 설명"""
    # 예시 영상 생성: 9x9 배열, 가운데 세로 방향의 흰색 띠
    original_fov = np.zeros((9, 9), dtype=float)
    original_fov[:, 3:6] = 1.0  # 세로 방향의 간단한 띠 패턴 생성
    angle_deg = 45.0

    """
    ndi.rotate() 함수는 이미지를 회전시킵니다.
    - angle=angle_deg: 회전 각도 지정
    - reshape=False: 입력과 출력의 배열 크기가 동일함 -> 원래 FOV만 표시
    - order=1: 선형 보간 사용
    - mode="constant": 경계값을 상수로 처리
    - cval=0.0: 경계값이 0.0
    - reshape=True: 회전된 영상이 잘리지 않도록 출력 영역이 확장됨
    """
    # reshape=False: 입력과 출력의 배열 크기가 동일함 -> 원래 FOV만 표시
    rotated_same_fov = ndi.rotate(
        original_fov, angle=angle_deg,
        reshape=False, order=1, mode="constant", cval=0.0
    )

    # reshape=True: 회전된 영상이 잘리지 않도록 출력 영역이 확장됨
    rotated_expanded = ndi.rotate(
        original_fov, angle=angle_deg,
        reshape=True, order=1, mode="constant", cval=0.0
    )

    print("[Rotation example]")
    print("Original FOV:       ", original_fov.shape)
    print("reshape=False:     ", rotated_same_fov.shape)
    print("reshape=True:      ", rotated_expanded.shape)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    examples = [
        (original_fov, "Original FOV"),
        (rotated_same_fov, "45° / reshape=False"),
        (rotated_expanded, "45° / reshape=True"),
    ]
    for ax, (img, title) in zip(axes, examples):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("ndi.rotate(): keep original FOV vs expand output")
    fig.tight_layout()
    plt.show()

show_rotation_example()

# %% [함수 1] 1D 주파수 필터 설계 ================================
def make_freq_filter_1d(length, filter_type="ramp", cutoff_ratio=0.5, order=2):
    """
    반환값
    -------
    freq_bins : shape (length,)   - fftshift된 주파수 인덱스
    H         : shape (length,)   - fftshift된 주파수 필터 응답

    필터 선택
    ---------
    ramp        : |f| 형태의 고역 통과 필터
    BPF         : ramp × Butterworth LPF (대역 통과 형태)

    * 필터 길이는 projection의 원소 수와 정확히 같아야 합니다.
    * 홀수/짝수 길이 모두 대응하도록 fftfreq()를 제공합니다.
    """

    # FFT 결과를 fftshift()로 옮겼을 때의 주파수 인덱스입니다.
    # 예: length=5 -> [-2, -1, 0, 1, 2]
    # 예: length=6 -> [-3, -2, -1, 0, 1, 2]
    freq_bins = np.fft.fftshift(np.fft.fftfreq(length)) * length # FFT 주파수 인덱스 생성 및 중앙 정렬
    nyquist_index = length / 2.0                                 # 나이퀴스트 주파수 인덱스 계산

    # TODO 1 [Ramp Filter]
    # Ramp Filter는 주파수의 절댓값에 비례하며,
    # 중심 주파수(0)에서 0이고 양쪽 고주파 방향으로 증가합니다.
    #
    # 수식: H(k) = |k| / (N/2)
    #
    # 힌트:
    # - freq_bins는 주파수 인덱스 배열입니다.
    # - np.abs()를 사용하여 각 주파수 인덱스의 절댓값을 계산하세요.
    # - nyquist_index로 나누어 필터의 최대값을 1로 정규화하세요.
    #
    # 계산 예시 (length = 8):
    # freq_bins     = [-4, -3, -2, -1, 0, 1, 2, 3]
    # nyquist_index = 4
    # ramp          = [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75]
        # 아래의 코드를 작성하세요.    
    ramp = np.abs(freq_bins) / nyquist_index

    if filter_type == "ramp":
        H = ramp
    else:
        # TODO 2 [ramp × Butterworth-windowed ramp]
        # 힌트: 
        # - cutoff = ... -> cutoff index = nyquist_index × cutoff_ratio
        # - butterworth_lpf = ... -> Butterworth LPF 식(실습 PDF 3쪽)을 적용합니다.
        # - H = ... -> 마지막으로 Ramp와 LPF를 곱해 최종 필터 H를 구하세요.
            # 아래의 코드를 작성하세요.    
        cutoff = nyquist_index * cutoff_ratio

        butterworth_lpf = 1 / (
        1 + (np.abs(freq_bins) / cutoff) ** (2 * order)
        )

        H = ramp * butterworth_lpf
        

    return freq_bins, H


# %% [함수 2] 각 projection의 1D frequency filtering ============
def apply_frequency_filter_1d(projection_1d, H):
    """
    입력 : projection_1d (1D), H (1D frequency response)
    출력 : filtered_projection_1d (1D)

    처리 순서
    1) FFT -> 2) fftshift -> 3) H를 곱함
    4) ifftshift -> 5) IFFT -> 6) 실수부
    """
    # 원래 Projection 길이
    original_length = projection_1d.size

    # Zero-padding
    padded_projection = np.pad(
        projection_1d,
        (0, H.size - original_length),
        mode="constant"
    )
    # TODO 3 [FFT + shift]
    # np.fft.fft(), np.fft.fftshift() 를 이용하세요.
        # 아래의 코드를 작성하세요.
    projection_fft = np.fft.fft(padded_projection)
    shifted_fft = np.fft.fftshift(projection_fft)


    # TODO 4 [Filtering + inverse FFT]
    # 주파수 영역에서 H를 곱하고, 위치를 되돌린 후 IFFT를 수행하세요. 
    # IFF결과는 복소수이므로 np.real()을 사용해 실수부만 취하세요.
        # 아래의 코드를 작성하세요.    
    filtered_fft = shifted_fft * H
    unshifted_fft = np.fft.ifftshift(filtered_fft)
    filtered_padded = np.real(np.fft.ifft(unshifted_fft))
    

    # 원래 Projection 길이로 복원
    filtered_projection_1d = filtered_padded[:original_length]

    return filtered_projection_1d


# %% [Problem 2] Frequency Filtering
def filter_sinogram(sinogram, H):
    num_angles = sinogram.shape[0]
    filtered_sinogram = np.zeros_like(sinogram, dtype=float)

    for angle_idx in range(num_angles):
        projection_1d = sinogram[angle_idx, :]
        filtered_sinogram[angle_idx, :] = apply_frequency_filter_1d(
            projection_1d, H
        )

    return filtered_sinogram


# %% [Problem 2] Backprojection
def backproject(filtered_sinogram):
    num_angles, num_detectors = filtered_sinogram.shape
    backprojected_images = []

    for angle_idx in range(num_angles):
        theta_deg = 180.0 * angle_idx / num_angles

        projection_2d = np.tile(
            filtered_sinogram[angle_idx, :], (num_detectors, 1)
        )

        rotated_projection = ndi.rotate(
            projection_2d,
            angle=theta_deg,
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0
        )

        backprojected_images.append(rotated_projection)

    return backprojected_images


# %% [Problem 2] Superposition
def superpose(backprojected_images):
    num_angles = len(backprojected_images)

    reconstructed_image = np.zeros_like(
        backprojected_images[0], dtype=float
    )

    for rotated_projection in backprojected_images:
        reconstructed_image += rotated_projection

    reconstructed_image = reconstructed_image / num_angles

    return reconstructed_image

# %% [FBP Main] 각도별 필터링 -> 역투영 -> 누적 ====================
def reconstruct_fbp(sinogram, filter_type="ramp", cutoff_ratio=0.5, order=2):
    """
    sinogram의 행 = 각도별 projection (0 <= theta < 180도)
    sinogram의 열 = detector 위치
    output: frequency axis, filter, filtered sinogram, reconstructed image
    """
    num_angles, num_detectors = sinogram.shape
    print(f"Number of projections (angles): {num_angles}")
    print(f"Number of detector bins:        {num_detectors}")

    # Zero-padding을 위한 FFT 길이 결정
    fft_length = 2 ** int(np.ceil(np.log2(2 * num_detectors - 1)))  # 512

    # (1) 한 개의 projection 길이와 동일한 주파수 필터 제작
    freq_bins, H = make_freq_filter_1d(
        length=fft_length,
        filter_type=filter_type,
        cutoff_ratio=cutoff_ratio,
        order=order,
    )

    # (2) 1D filtering: 각 행(각도)의 projection에 동일한 H를 적용
    filtered_sinogram = np.zeros_like(sinogram, dtype=float)

    for angle_idx in range(num_angles):
        projection_1d = sinogram[angle_idx, :]

        # TODO 5 [Apply filter to each projection]
        # 위에서 작성한 apply_frequency_filter_1d() 함수를 호출하세요.
        # 결과를 filtered_sinogram의 해당 행에 저장하세요.
            # 아래의 코드를 작성하세요.    
        filtered_projection_1d = apply_frequency_filter_1d(
        projection_1d, H)
        filtered_sinogram[angle_idx, :] = filtered_projection_1d
        

    # (3) FOV: detector 수만큼 정사각형으로 복원 영역 설정
    # 회전 후에도 같은 크기를 유지하기 위해 reshape=False를 사용합니다.
    reconstructed_image = np.zeros((num_detectors, num_detectors), dtype=float)

    # (4) Backprojection + (5) Superposition
    for angle_idx in range(num_angles):
        # 0 <= theta < 180°, 마지막 각도는 180° 미만입니다.
        theta_deg = 180.0 * angle_idx / num_angles

        # 한 줄의 projection 값을 세로 방향으로 반복하여 2D로 만듭니다.
        # 예: (N,) -> (N, N). 
        projection_2d = np.tile(filtered_sinogram[angle_idx, :], (num_detectors, 1))

        # TODO 6 [Backprojection: rotation]
        # 예제 show_rotation_example()을 참고하세요.
        # scipy.ndimage.rotate()를 사용해 theta_deg만큼 회전시키세요.
        # reshape=False (FOV 유지), order=1 (선형 보간)을 지정하세요.
        # * 원본 sinogram의 각도 정의에 따라 angle 부호를 확인하세요.
            # 아래의 코드를 작성하세요.    
    
        rotated_projection = ndi.rotate(
        projection_2d,
        angle=theta_deg,
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0
        )

        # TODO 7 [Superposition]
        # 매 각도에서 얻은 rotated_projection을 reconstructed_image에 누적하세요.
            # 아래의 코드를 작성하세요.
        # reconstructed_image ...

        reconstructed_image += rotated_projection

    # TODO 8 [Normalization]
    # 각도별 결과를 모두 더했으므로 projection 개수를 고려해 정규화하세요.(힌트: num_angles)
        # 아래의 코드를 작성하세요.    
    # reconstructed_image...
    reconstructed_image = reconstructed_image / num_angles

    return freq_bins, H, filtered_sinogram, reconstructed_image

# %% [Problem 2] FBP using separate functions
def reconstruct_fbp_functions(sinogram, filter_type="ramp", cutoff_ratio=0.5, order=2):

    num_angles, num_detectors = sinogram.shape

    # (1) Frequency Filter
    fft_length = 2 ** int(np.ceil(np.log2(2 * num_detectors - 1)))

    freq_bins, H = make_freq_filter_1d(
        length=fft_length,
        filter_type=filter_type,
        cutoff_ratio=cutoff_ratio,
        order=order,
    )

    # (2) Frequency Filtering
    filtered_sinogram = filter_sinogram(sinogram, H)

    # (3) Backprojection
    backprojected_images = backproject(filtered_sinogram)

    # (4) Superposition
    reconstructed_image = superpose(backprojected_images)

    return freq_bins, H, filtered_sinogram, reconstructed_image


# %% [시각화] (수정 불필요) ========================
def show_results(sinogram, freq_bins, H, filtered_sinogram, reconstructed_image):
    """원본 sinogram -> 주파수 필터 -> filtered sinogram -> 복원 영상."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    axes[0, 0].imshow(sinogram, cmap="gray", aspect="auto")
    axes[0, 0].set_title("Original Sinogram")
    axes[0, 0].set_xlabel("Detector bin")
    axes[0, 0].set_ylabel("Projection index")

    axes[0, 1].plot(freq_bins, H)
    axes[0, 1].set_title("Frequency Filter Response")
    axes[0, 1].set_xlabel("Frequency index")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].imshow(filtered_sinogram, cmap="gray", aspect="auto")
    axes[1, 0].set_title("Filtered Sinogram")
    axes[1, 0].set_xlabel("Detector bin")
    axes[1, 0].set_ylabel("Projection index")

    axes[1, 1].imshow(reconstructed_image, cmap="gray")
    axes[1, 1].set_title("FBP Reconstruction")
    axes[1, 1].axis("off")

    fig.tight_layout()
    plt.show()


# %% [설정] =======================================================
# 아래 이름과 parameter를 변경하여 실험하세요.
INPUT_FILENAME = "sinogram_lownoise.png"
# INPUT_FILENAME = "sinogram_highnoise.png"
# INPUT_FILENAME = "sinogram_lownoise.png"

FILTER_TYPE = "BPF"           # "ramp" 또는 "BPF"
CUTOFF_RATIO = 0.3             # BPF cutoff: 0.1 ~ 1.0 (Nyquist 기준 비율)
FILTER_ORDER = 4               # BPF order: 2부터 실험
SHOW_ROTATION_EXAMPLE = False   # 회전 예제를 먼저 볼지 선택


# %% [실행] =======================================================
def main():
    # 원본 영상 파일은 이 코드를 실행하는 '현재 작업 폴더'에 두세요.
    # 지정한 파일이 없으면 FileNotFoundError가 발생합니다.
    sinogram = io.imread(INPUT_FILENAME)
    sinogram = np.asarray(sinogram, dtype=float)


    freq_bins, H, filtered_sinogram, reconstructed_image = reconstruct_fbp_functions(
        sinogram,
        filter_type=FILTER_TYPE,
        cutoff_ratio=CUTOFF_RATIO,
        order=FILTER_ORDER,
    )

    # 시각화
    show_results(sinogram, freq_bins, H, filtered_sinogram, reconstructed_image)


if __name__ == "__main__":
    main()

# %%
