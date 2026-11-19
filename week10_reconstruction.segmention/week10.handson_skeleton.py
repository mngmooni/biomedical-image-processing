# %% BMED318 Week10: Image Restoration & Segmentation
# 강의 자료, 실습 문제와 결과 예시를 참고하여 TODO의 None/[]를 채우세요.
# 미작성 상태의 계산 오류는 그대로 표시됩니다. 1번은 1-0 셀에서 먼저 좌표를 확인하세요.
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature, transform

DPATH = Path(__file__).resolve().parent          # 이미지가 있는 폴더


def show(image, title, **kwargs):                # 회색조 영상과 제목 표시
    plt.imshow(image, cmap='gray', **kwargs)
    plt.title(title)
    plt.axis('off')


# %% 1-0. 원본과 스펙트럼 확인
image = io.imread(DPATH / 'lena_noise.png').astype(float)
G = np.fft.fftshift(np.fft.fft2(image))           # 영상의 FFT를 구하고 DC 성분을 중앙으로 정렬

plt.figure(figsize=(9, 4))
plt.subplot(121); show(image, 'Input', vmin=0, vmax=255)
plt.subplot(122); show(np.log1p(np.abs(G)), 'Spectrum: find (x, y)')  # 스펙트럼 크기를 로그로 압축해 작은 성분도 표시
plt.axis('on')                                  # Figure 창의 도구 모음에서 마우스 좌표 확인
plt.tight_layout()
plt.show()


# %% 1. Periodic noise: notch filter (1-0 셀 실행 후 작성) - ch8.Image restoration.pdf page 24-28
yy, xx = np.indices(image.shape)                # 각 픽셀의 행 좌표와 열 좌표를 배열로 생성
noise_points = [   (47, 87),
    (87, 87),
    (169, 169),
    (209, 169)]               # TODO 1-1: 잡음 좌표 (x, y)를 중심 대칭 쌍으로 선택
radius = 5                                      # 제거할 원형 영역의 반지름

notch_mask = np.ones(image.shape)                # 모든 주파수를 통과시키도록 마스크를 1로 초기화
for x, y in noise_points:  # 선택한 잡음점을 하나씩 중심으로 삼아 제거 영역 계산
    dx = xx - x                                 # 각 픽셀과 잡음점 사이의 가로 좌표 차이
    dy = yy - y                                 # 각 픽셀과 잡음점 사이의 세로 좌표 차이
    distance2 = dx**2 + dy**2              # TODO 1-2: 각 픽셀과 잡음점 사이의 거리 제곱 계산
    notch_mask[distance2 <= radius**2] = 0       # 잡음점에서 반지름 이내인 주파수 성분을 차단

filtered_spectrum =  G * notch_mask           # TODO 1-3a: 입력 스펙트럼에 notch 마스크 적용
notch_result = np.fft.ifft2(
    np.fft.ifftshift(filtered_spectrum)
).real                # TODO 1-3b: 주파수 배치를 되돌리고 역FFT로 영상 복원

plt.figure(figsize=(9, 8))
plt.subplot(221); show(image, 'Input', vmin=0, vmax=255)
plt.subplot(222); show(np.log1p(np.abs(G)), 'Spectrum: find (x, y)')
plt.axis('on')
xs, ys = np.array(noise_points).T                 # 선택한 잡음 좌표 표시
plt.scatter(xs, ys, c='red', s=20)
plt.subplot(223); show(notch_mask, 'Notch mask', vmin=0, vmax=1)
plt.subplot(224); show(notch_result, 'Restored image', vmin=0, vmax=255)
plt.tight_layout()
plt.show()


# %% 2. Degraded image: inverse / Wiener filter - ch8.Image restoration.pdf page 29-36, 43-47
image = io.imread(DPATH / 'keyboard.png').astype(float)
G = np.fft.fftshift(np.fft.fft2(image))  # 영상의 FFT를 구하고 DC 성분을 중앙으로 정렬

size, sigma = 15, 0.9                            # 문제에서 주어진 열화 조건
q = np.arange(size) - size // 2  # 가우시안 중심을 0으로 둔 1차원 좌표 생성
xx, yy = np.meshgrid(q, q)  # 가우시안 커널 계산용 2차원 좌표 생성
kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))  # 중심에서 멀어질수록 작아지는 가우시안 가중치 계산
kernel /= kernel.sum()  # 커널 합을 1로 정규화해 밝기 기준 유지
h = np.zeros(image.shape)  # 커널을 담을 영상 크기의 배열 준비
h[:size, :size] = kernel  # 작은 가우시안 커널을 영상 크기의 배열에 배치
h = np.roll(h, (-(size // 2), -(size // 2)), axis=(0, 1))  # 커널 중심을 FFT 원점으로 옮겨 영상 위치 이동 방지
H = np.fft.fftshift(np.fft.fft2(h))               # 열화 커널의 주파수 응답 계산

inverse_response = 1.0 / H            # TODO 2-1: 열화 응답을 상쇄하는 역필터 응답 계산
inverse_spectrum = G * inverse_response  # 입력 스펙트럼에 역필터 응답 적용
inverse_result = np.fft.ifft2(
    np.fft.ifftshift(inverse_spectrum)
).real              # TODO 2-2: 역필터 적용 결과를 역변환해 실수 영상으로 복원

K = 0.01                                        # Wiener의 잡음 증폭 억제 상수로 0.01~1.0에서 비교
H_power = np.abs(H)**2  # Wiener 필터에 사용할 열화 응답의 크기 제곱 계산
wiener_response = np.conj(H) / (H_power + K)             # TODO 2-3: 열화 보상과 잡음 증폭 억제를 위한 Wiener 응답 계산
wiener_spectrum = G * wiener_response  # 입력 스펙트럼에 Wiener 응답 적용
wiener_result = np.fft.ifft2(
    np.fft.ifftshift(wiener_spectrum)
).real               # TODO 2-4: Wiener 적용 결과를 역변환해 실수 영상으로 복원

plt.figure(figsize=(12, 8))
plt.subplot(231); show(image, 'Input', vmin=0, vmax=255)
plt.subplot(232); show(inverse_result, 'Inverse filter', vmin=0, vmax=255)
plt.subplot(233); show(wiener_result, f'Wiener filter: K={K}', vmin=0, vmax=255)
plt.subplot(234); show(np.log1p(np.abs(G)), 'Input spectrum')
plt.subplot(235); show(np.log1p(np.abs(H)), 'Gaussian spectrum')
plt.subplot(236)
mid = image.shape[0] // 2                        # 중앙 행의 주파수 응답 비교
freq = np.arange(image.shape[1]) - image.shape[1] // 2  # DC를 0으로 둔 주파수 인덱스 생성
plt.semilogy(freq, np.abs(H[mid]), label='Gaussian')
plt.semilogy(freq, np.abs(inverse_response[mid]), label='Inverse')
plt.semilogy(freq, np.abs(wiener_response[mid]), label='Wiener')
plt.title('Center-row filter responses')
plt.xlabel('Frequency index'); plt.ylabel('Magnitude (log scale)')
plt.legend()
plt.tight_layout()
plt.show()


# %% 3. Binary segmentation: p1 / p2 - ch9.Image segmentation.pdf page 12-17
for filename in ['p1.jpg', 'p2.jpg']:
    image = io.imread(DPATH / filename)
    rows, cols = image.shape                    # 영상의 높이와 너비
    # 조명과 물체 위치를 보고 처리 영역의 개수·크기·배치를 직접 설계하세요.
    # 영역 형식: (r0, r1, c0, c1) = (시작 행, 끝 행, 시작 열, 끝 열), 끝은 미포함
    # 세로 구간은 행 범위를 0~rows로, 사각 블록은 행·열 범위를 함께 지정합니다.
    if filename == 'p1.jpg':
        col_edges = np.linspace(0, cols, 11, dtype=int)
        regions = [(0, rows, col_edges[i], col_edges[i + 1])
        for i in range(10)]              # TODO 3-0a: p1의 처리 영역 목록 설계
    else:
        row_edges = np.linspace(0, rows, 4, dtype=int)
        col_edges = np.linspace(0, cols, 7, dtype=int)
        regions = [(row_edges[i], row_edges[i + 1],
         col_edges[j], col_edges[j + 1])
        for i in range(3)
        for j in range(6)]              # TODO 3-0b: p2의 처리 영역 목록 설계

    binary = np.zeros(image.shape, dtype=bool)    # 선택하지 않은 영역의 초기값은 False
    for r0, r1, c0, c1 in regions:
        block = image[r0:r1, c0:c1]              # 선택한 사각 영역의 픽셀 추출
        threshold = filters.threshold_otsu(block)          # TODO 3-1: 현재 영역의 Otsu 임계값 계산
        if filename == 'p1.jpg':
            block_binary =  block > threshold  # TODO 3-2: 밝은 물체가 True인 영역 영상 생성
        else:
            block_binary = block < threshold   # TODO 3-3: 어두운 글자가 True인 영역 영상 생성
        binary[r0:r1, c0:c1] = block_binary      # 같은 위치에 영역별 분할 결과 저장

    plt.figure(figsize=(12, 4))
    plt.suptitle(f'Problem 3: {filename}')
    plt.subplot(131); show(image, 'Input', vmin=0, vmax=255)
    plt.subplot(132)
    plt.hist(image.ravel(), bins=256, range=(0, 256))
    plt.title('Histogram'); plt.xlabel('Gray level')
    plt.subplot(133); show(binary, 'Binary segmentation', vmin=0, vmax=1)
    plt.tight_layout()
    plt.show()


# %% 4. Edge segmentation: Canny - ch9.Image segmentation.pdf page 26-30
image = io.imread(DPATH / 'p3.jpg')
image_float = image / 255.0                      # 픽셀값을 0~1로 맞춰 Canny 임계값과 단위 통일
settings = [(1, 0.1, 0.2),
    (2, 0.1, 0.2),
    (3, 0.1, 0.2)]  # sigma, low, high

plt.figure(figsize=(14, 4))
plt.subplot(1, len(settings) + 1, 1); show(image, 'Input', vmin=0, vmax=255)
for i, (sigma, low, high) in enumerate(settings):
    edges = feature.canny(
    image_float,
    sigma=sigma,
    low_threshold=low,
    high_threshold=high
    )                  # TODO 4: 평활화 후 밝기 변화가 큰 경계 검출
    plt.subplot(1, len(settings) + 1, i + 2)
    show(edges, f'sigma={sigma}\nlow={low}, high={high}', vmin=0, vmax=1)
plt.tight_layout()
plt.show()


# %% 5. Dominant edges: Hough transform - ch9.Image segmentation.pdf page 31-38
sigma, low, high = 2, 0.1, 0.2
num_peaks = 10                                  # 주요 직선의 최대 개수

for filename in ['p3.jpg', 'p4.jpg', 'p5.jpg', 'p6.jpg']:
    image = io.imread(DPATH / filename)
    image_float = image / 255.0  # 픽셀값을 0~1로 맞춰 Canny 임계값과 단위 통일
    edges = feature.canny(
    image_float,
    sigma=sigma,
    low_threshold=low,
    high_threshold=high
    )                  # TODO 5-1: 직선 검출에 사용할 에지 영상 계산
    hspace, angles, distances = transform.hough_line(edges) # TODO 5-2: 에지 픽셀의 각도·거리별 직선 투표 누적
    peak_threshold = 0.2 * hspace.max()  # 최대 투표수의 20%를 주요 직선 선택 기준으로 사용
    peaks, line_angles, line_distances = transform.hough_line_peaks(
    hspace,
    angles,
    distances,
    num_peaks=num_peaks,
    threshold=peak_threshold
    )  # TODO 5-3: 투표수가 큰 주요 직선 후보를 최대 num_peaks개 선택

    plt.figure(figsize=(14, 4))
    plt.suptitle(f'Problem 5: {filename}')
    plt.subplot(141); show(image, 'Input', vmin=0, vmax=255)
    plt.subplot(142); show(edges, 'Edges', vmin=0, vmax=1)
    plt.subplot(143)
    plt.imshow(np.log1p(hspace), cmap='gray', aspect='auto', origin='lower',
               extent=[np.rad2deg(angles[0]), np.rad2deg(angles[-1]),
                       distances[0], distances[-1]])
    plt.title('Hough accumulator')
    plt.xlabel('Theta (degree)'); plt.ylabel('Rho (pixel)')
    plt.subplot(144); show(image, 'Detected lines', vmin=0, vmax=255)
    for theta, rho in zip(line_angles, line_distances):
        x0, y0 = rho * np.cos(theta), rho * np.sin(theta)  # 검출 직선 위에서 원점에 가장 가까운 기준점 계산
        plt.axline((x0, y0), (x0 - np.sin(theta), y0 + np.cos(theta)),
                   color='red', linewidth=1)
    plt.xlim(0, image.shape[1] - 1)
    plt.ylim(image.shape[0] - 1, 0)
    plt.tight_layout()
    plt.show()
