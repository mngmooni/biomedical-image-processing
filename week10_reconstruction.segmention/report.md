## 1.	Restoration from periodic noise using notch filter
   
### 1.1.	구현 방법 (Method)
주기적 잡음이 포함된 lena_noise.png를 복원하기 위해 frequency domain에서 Ideal Notch Reject Filter를 적용하였다.
먼저 np.fft.fft2()를 이용하여 입력 영상에 2D Fourier Transform을 수행하고, np.fft.fftshift()를 통해 DC component를 spectrum 중앙으로 이동시켰다. 이후 np.log1p(np.abs(G))로 magnitude spectrum을 시각화하여 주기적 잡음에 해당하는 밝은 peak의 위치를 확인하였다.
Spectrum에서 확인한 네 개의 noise peak를 (47, 87), (87, 87), (169, 169), (209, 169)로 설정하였다. 이 좌표들은 중심에 대해 대칭적으로 나타나는 주기적 잡음 성분에 해당한다. Notch mask는 모든 값을 1로 초기화한 후, 각 noise peak를 중심으로 반지름 radius 이내의 영역을 0으로 설정하는 방식으로 구현하였다. 이를 위해 np.indices()로 좌표 배열을 생성하고, 각 픽셀과 noise peak 사이의 거리 제곱을 계산하였다.

d² = (x − x₀)² + (y − y₀)²

distance2 <= radius**2 조건을 만족하는 영역을 0으로 설정하여 해당 주파수 성분을 제거하였다.
이후 filtered_spectrum = G * notch_mask로 frequency filtering을 수행하고, np.fft.ifftshift()와 np.fft.ifft2()를 이용하여 spatial domain의 복원 영상을 얻었다.

### 1.2.	Radius 비교 및 선택

Notch Filter의 반지름에 따른 복원 결과를 비교하기 위해 radius = 3, 5, 8을 각각 적용하였다.
•	Radius = 3: 주요 periodic noise는 감소하였으나, 영상 전반에 미세한 대각선 줄무늬가 남아 있었다. 제거하는 주파수 영역이 상대적으로 작아 영상 정보의 손실은 적지만 잡음 억제 효과가 제한적이었다.
•	Radius = 5: Radius 3에 비해 잔여 줄무늬가 다소 감소하였으며, Lena의 얼굴과 모자 등 주요 구조도 비교적 잘 유지되었다. 잡음 제거와 영상의 세부 정보 보존 사이에서 적절한 균형을 보였다.
•	Radius = 8: Notch mask의 차단 영역이 더 넓어졌지만, Radius 5 대비 추가적인 시각적 개선은 크지 않았다. 반면 차단 영역이 커질수록 잡음 이외의 유용한 주파수 성분까지 제거할 가능성이 증가한다.
따라서 최종적으로 radius = 5를 선택하였다. 세 가지 실험 결과에서 Radius 5는 Radius 3보다 잡음 억제 효과가 다소 좋았으며, Radius 8과 비교했을 때 영상 복원의 추가적인 이득은 크지 않으면서 더 적은 주파수 영역을 제거하였다.

<img width="438" height="426" alt="image" src="https://github.com/user-attachments/assets/360724e5-17ef-4d0b-8b48-12a938f2300e" />
[최종 선택: Radius=5]


### 1.3. 결과 및 고찰

Notch Reject Filtering을 적용한 결과, 원본 영상에 나타나던 강한 periodic stripe pattern이 감소하여 Lena의 주요 구조를 보다 명확하게 확인할 수 있었다.
다만 복원 영상에 일부 미세한 줄무늬가 남아 있었으며, radius를 증가시키는 것만으로는 이를 완전히 제거하기 어려웠다. 이는 선택한 네 개의 noise peak 주변만을 차단하는 현재 필터의 한계를 보여준다.
결과적으로 Notch Filter는 특정 주파수에 집중된 periodic noise를 선택적으로 제거하는 데 효과적이며, radius는 잡음 억제와 영상 정보 보존을 함께 고려하여 결정해야 함을 확인하였다.

## 2.	Restoration of degraded image

### 2.1.	구현 방법 (Method)
Gaussian Blur로 열화된 keyboard.png를 복원하기 위해 Inverse Filter와 Wiener Filter를 각각 구현하였다.
먼저 np.fft.fft2()와 np.fft.fftshift()를 이용하여 입력 영상을 frequency domain으로 변환하였다. 문제에서 주어진 Gaussian Filter의 조건인 size=(15, 15), sigma=0.9를 사용하여 Gaussian kernel을 생성하고, 전체 영상 크기에 맞게 zero-padding한 뒤 np.roll()로 kernel의 중심을 FFT 원점으로 이동시켰다. 이후 Fourier Transform을 적용하여 degradation function (H(u,v))를 계산하였다.
Inverse Filter는 degradation function의 역수를 이용하는 방식으로 구현하였다.

W_inv(u, v) = 1 / H(u, v)

코드에서는 inverse_response = 1.0 / H로 filter response를 계산하고, 이를 입력 영상의 spectrum에 곱하였다. 이후 np.fft.ifftshift()와 np.fft.ifft2()를 이용하여 복원 영상을 얻었다.
Wiener Filter는 잡음 증폭을 억제하기 위해 다음 식을 사용하였다.

W_Wiener(u, v) = H*(u, v) / (|H(u, v)|² + K)

np.conj(H)로 complex conjugate를 계산하고, np.abs(H)**2로 magnitude squared를 구하였다. 이를 이용하여 wiener_response = np.conj(H) / (H_power + K)를 구현한 후, Inverse FFT를 통해 복원 영상을 얻었다.

### 2.2.	Inverse Filter와 Wiener Filter 결과 비교

Inverse Filter를 적용한 결과, 입력 영상에 비해 키보드의 글자와 경계가 선명해졌으나, 어두운 배경과 키보드 주변에서 잡음이 크게 증폭되는 현상이 관찰되었다. 이는 degradation function (H(u,v))가 작은 고주파 영역에서 역필터의 응답이 급격히 증가하기 때문이다.
반면 Wiener Filter는 고주파 성분의 과도한 증폭을 억제하여 Inverse Filter보다 잡음이 적은 복원 결과를 보였다. 특히 K=0.01에서는 키보드의 주요 구조와 밝기를 비교적 잘 유지하면서 잡음 증폭을 제한할 수 있었다.
Frequency Response를 log scale로 비교한 결과, Gaussian Filter는 고주파에서 응답이 감소하는 Low-Pass 특성, Inverse Filter는 고주파 응답이 증가하는 High-Pass 특성을 나타냈다. Wiener Filter는 중간 주파수 영역의 복원 효과를 유지하면서 높은 주파수의 과도한 증폭을 억제하는 특성을 보였다.

### 2.3.	K 값 비교 및 선택

Wiener Filter의 파라미터 (K)가 복원 결과에 미치는 영향을 확인하기 위해 K = 0.01, 0.1, 1.0을 각각 적용하였다.
•	K = 0.01: 키보드의 글자와 경계가 비교적 선명하게 복원되었으며, 입력 영상의 밝기도 잘 유지되었다. 고주파 증폭이 상대적으로 컸지만 Inverse Filter에서 나타난 심한 잡음 증폭은 억제되었다.
•	K = 0.1: K=0.01에 비해 고주파 증폭이 감소하였다. 복원 영상은 상대적으로 부드러워졌으며, 일부 세부 정보와 밝기가 감소하였다.
•	K = 1.0: Frequency Response가 전반적으로 크게 감소하였고, 복원 영상이 눈에 띄게 어두워졌다. 잡음 증폭은 억제되었지만 영상의 밝기와 세부 구조까지 약화되어 복원 효과가 제한적이었다.
따라서 최종적으로 K=0.01을 선택하였다. 실험한 세 값 중 원본의 밝기와 주요 구조를 가장 잘 유지하면서 Inverse Filter보다 잡음 증폭을 효과적으로 억제하였기 때문이다.

<img width="940" height="627" alt="image" src="https://github.com/user-attachments/assets/559517d2-eea4-4a67-bb32-262ad80d1b78" />

[최종 선택: k=0.01]


### 2.4.	결과 및 고찰

Inverse Filter와 Wiener Filter 모두 Gaussian Blur에 의해 열화된 영상의 복원에 사용될 수 있지만, 잡음에 대한 반응에서는 차이가 나타났다.
Inverse Filter는 degradation function을 직접 역으로 보정하므로 고주파 성분을 강하게 복원하지만, 이 과정에서 잡음까지 증폭시키는 한계가 있었다. Wiener Filter는 (K)를 이용하여 이러한 증폭을 제한함으로써 상대적으로 안정적인 복원 결과를 제공하였다.
또한 K가 증가할수록 잡음 억제는 강해지지만, 동시에 영상의 밝기와 세부 정보도 감소할 수 있음을 확인하였다. 따라서 Wiener Filtering에서는 잡음 억제와 영상 정보 보존 사이의 균형을 고려한 K 선택이 중요함을 확인하였다.

## 3.	Binary Segmentation

### 3.1.	구현 방법 (Method)
불균일한 조명을 가진 p1.jpg와 p2.jpg를 Binary Segmentation하기 위해 Block-wise Adaptive Thresholding을 적용하였다.
먼저 각 영상의 Histogram을 확인하고, 영상 전체에 하나의 Threshold를 적용하는 대신 여러 Block으로 분할하여 각 영역에 독립적인 Otsu Threshold를 계산하였다. 이는 조명 변화가 큰 영상에서 Global Thresholding의 한계를 보완하기 위한 방법이다.
np.linspace()를 이용하여 영상의 행과 열에 대한 분할 경계를 생성하고, 각 Block을 (r0, r1, c0, c1) 형식의 regions로 정의하였다. 이후 filters.threshold_otsu(block)을 이용하여 각 영역의 Threshold를 계산하였다.
두 영상의 분할 대상이 다르므로 Threshold 적용 방향도 구분하였다.
•	p1.jpg: 밝은 물체를 추출하기 위해 block > threshold를 적용하였다.
•	p2.jpg: 어두운 글자를 추출하기 위해 block < threshold를 적용하였다.
각 Block의 Binary Segmentation 결과를 원래 영상의 해당 위치에 저장하여 최종 Binary Image를 생성하였다.

### 3.2.	p1.jpg — Block 분할 결과 비교

<img width="921" height="328" alt="image" src="https://github.com/user-attachments/assets/ec469d0f-fa58-4702-b6cb-49c31f86f9f2" />

p1 영상은 왼쪽에서 오른쪽으로 배경의 밝기가 증가하는 특징을 가지고 있어, 세로 방향으로 Block을 분할하였다.
먼저 세로 6등분과 10등분을 비교하였다. 두 방법 모두 중앙의 밝은 물체들을 비교적 정확하게 분할하였으며, 물체의 형태와 윤곽도 잘 유지되었다.
그러나 양쪽 가장자리에서 배경 일부가 흰색으로 잘못 분류되는 현상이 관찰되었다. 특히 Block 개수를 10개로 늘려도 이러한 오검출이 뚜렷하게 개선되지는 않았다.
최종적으로 세로 10등분을 사용하였다. 조명 변화에 따라 Threshold를 더 세밀하게 계산하면서도 주요 물체의 형태를 유지할 수 있었기 때문이다. 다만 6등분에 비해 개선 폭은 크지 않았다.

### 3.3.	p2.jpg — Block 분할 방식 비교

<img width="935" height="312" alt="image" src="https://github.com/user-attachments/assets/4ee9685e-d948-4546-a05c-3df6c125821a" />

p2 영상은 조명이 불균일한 문서 영상으로, 어두운 글자와 밝은 배경을 구분하기 위해 다양한 Block 분할 방식을 실험하였다.
•	세로 6등분: 글자 대부분이 검출되었으나, 중앙에 넓은 배경 오검출 영역이 발생하였다.
•	세로 10등분: 일부 영역에서 오검출이 감소했지만, 세로 방향의 불균일한 분할이 여전히 나타났다.
•	2행 × 5열: 왼쪽 영역의 글자 검출이 개선되었으나, 중앙의 넓은 흰색 오검출 영역이 남아 있었다.
•	4행 × 8열: 글자 검출은 비교적 양호했지만, 오른쪽 아래에서 반복적인 흰색 영역이 발생하여 배경 오검출이 증가하였다.
•	3행 × 6열: 글자 검출을 비교적 잘 유지하면서 4행 × 8열에서 나타났던 오른쪽 아래의 오검출이 감소하였다.
따라서 최종적으로 3행 × 6열, 총 18개 Block을 선택하였다. 실험한 분할 방식 중 글자 검출과 배경 오검출 사이에서 비교적 적절한 균형을 보였기 때문이다.


<img width="940" height="335" alt="image" src="https://github.com/user-attachments/assets/eb795dd0-d4cc-41f2-8ecf-3860ee6c5dc6" />
<img width="940" height="314" alt="image" src="https://github.com/user-attachments/assets/7c1238c0-b03f-4672-8a0a-8e6d6bab320d" />
[최종 선택 p1: 10 vertical blocks, p2: 3 x 6 blocks]



### 3.4.	결과 및 고찰

Block-wise Adaptive Thresholding을 적용하여 두 영상의 주요 물체와 글자를 Binary Image로 추출할 수 있었다.
p1에서는 밝은 물체의 윤곽이 비교적 정확하게 보존되었으나, 양쪽 가장자리에서 일부 배경 오검출이 발생하였다. p2에서는 Block의 개수와 배치에 따라 글자 검출 정도와 배경 오검출 양상이 달라졌다.
특히 Block 크기를 지나치게 줄이는 것이 반드시 좋은 결과로 이어지지는 않았다. 각 Block의 Histogram에서 물체와 배경의 intensity distribution이 충분히 구분되지 않으면 Otsu Thresholding이 배경을 잘못 분류할 수 있기 때문이다.
결과적으로 Adaptive Thresholding의 성능은 Block 개수뿐 아니라 크기와 배치에도 영향을 받으며, 영상의 조명 분포와 분할 대상의 특성을 고려한 영역 설정이 중요함을 확인하였다.

## 4.	Edge Segmentation

### 4.1.	구현 방법 (Method)

p3.jpg에서 건물의 주요 경계를 검출하기 위해 Canny Edge Detector를 적용하였다.
먼저 입력 영상의 pixel intensity를 255로 나누어 0~1 범위로 정규화하였다. 이후 skimage.feature.canny() 함수를 사용하여 Edge Segmentation을 수행하였다.
Canny Edge Detector는 Gaussian Smoothing으로 잡음을 억제한 후, gradient를 이용하여 경계 후보를 검출한다. 이어서 Non-maxima Suppression으로 경계를 가늘게 만들고, Hysteresis Thresholding을 통해 강한 Edge와 이에 연결된 약한 Edge를 최종적으로 선택한다.
코드에서는 feature.canny(image_float, sigma=sigma, low_threshold=low, high_threshold=high)를 사용하였으며, Gaussian Smoothing의 강도를 결정하는 sigma와 Hysteresis Thresholding에 사용하는 low_threshold, high_threshold를 변경하며 결과를 비교하였다.

### 4.2.	Sigma에 따른 결과 비교

<img width="940" height="210" alt="image" src="https://github.com/user-attachments/assets/4a37c221-2c1c-4ba4-81bc-b3f27ee36b55" />


먼저 low_threshold=0.1, high_threshold=0.2로 고정하고 sigma=1, 2, 3을 비교하였다.
•	Sigma = 1: Gaussian Smoothing이 상대적으로 약해 지붕의 세밀한 패턴과 벽돌의 질감까지 Edge로 검출되었다. 많은 세부 정보를 보존했지만 불필요한 Edge도 다수 포함되었다.
•	Sigma = 2: 벽면의 불필요한 Edge가 감소하면서 건물의 지붕, 외벽, 창문 등의 주요 경계가 비교적 잘 유지되었다.
•	Sigma = 3: 강한 Smoothing으로 Edge 영상이 더욱 단순해졌으나, 지붕과 창문 등 일부 구조의 세부 경계가 약해지거나 사라졌다.
따라서 sigma=2가 주요 구조의 경계 보존과 불필요한 Edge 억제 사이에서 가장 적절한 균형을 보였다.


### 4.3.	Threshold에 따른 결과 비교

<img width="940" height="210" alt="image" src="https://github.com/user-attachments/assets/73593b75-73b1-4eb5-a6bd-cf1206480c16" />

다음으로 sigma=2를 고정하고 세 가지 Hysteresis Threshold 조합을 비교하였다.
•	Low = 0.05, High = 0.15: 낮은 Threshold로 인해 약한 Edge까지 검출되면서 건물 벽면과 지붕에서 불필요한 선이 증가하였다.
•	Low = 0.1, High = 0.2: 건물의 주요 윤곽과 창문 경계가 비교적 연속적으로 유지되었으며, 불필요한 Edge도 효과적으로 억제되었다.
•	Low = 0.2, High = 0.3: 강한 Edge를 중심으로 검출되어 결과가 단순해졌으나, 지붕과 창문의 일부 경계가 끊어지거나 사라졌다.

Threshold가 낮을수록 약한 경계까지 검출되지만 불필요한 Edge가 증가할 수 있으며, Threshold가 높을수록 잡음은 감소하지만 실제 경계까지 누락될 수 있음을 확인하였다.

### 4.4.	최종 파라미터 선택 및 고찰

<img width="940" height="210" alt="image" src="https://github.com/user-attachments/assets/4044b962-1373-410b-bf35-ac60971a1083" />
[최종 선택: sigma=2, low=0.1, high=0.2]

실험 결과를 종합하여 sigma=2, low_threshold=0.1, high_threshold=0.2를 최종 설정으로 선택하였다.
해당 설정은 sigma=1보다 불필요한 세부 Edge를 적게 검출하면서도 sigma=3보다 주요 구조의 경계를 잘 보존하였다. 또한 Threshold 비교에서도 건물의 주요 윤곽을 유지하면서 과도한 Edge 검출을 방지하는 결과를 보였다.
이를 통해 Canny Edge Detection에서는 Gaussian Smoothing의 강도와 Hysteresis Threshold를 함께 조정하는 것이 중요함을 확인하였다. 지나치게 약한 Smoothing이나 낮은 Threshold는 불필요한 Edge를 증가시키고, 반대로 강한 Smoothing이나 높은 Threshold는 필요한 경계를 제거할 수 있다.

## 5.	Segmentation of Dominant Edges (Hough/Radon Transform)

### 5.1. 구현 방법 (Method)

p3.jpg부터 p6.jpg까지 총 4개의 영상에서 주요 직선 경계를 검출하기 위해 Canny Edge Detector와 Hough Transform을 적용하였다.
먼저 입력 영상의 pixel intensity를 0~1 범위로 정규화하고, feature.canny()를 이용하여 Binary Edge Image를 생성하였다. Canny의 파라미터는 앞선 실험을 참고하여 sigma=2, low_threshold=0.1, high_threshold=0.2로 설정하였다.
이후 transform.hough_line(edges)를 이용하여 검출된 Edge 픽셀을 Hough parameter space로 변환하였다. Hough Transform은 다음 직선 방정식을 기반으로 한다.
[ρ = x cos(θ) + y sin(θ)]
각 Edge 픽셀에 대해 가능한 (ρ, θ) 조합의 투표를 누적하여 Hough Accumulator를 생성하였다.
다음으로 transform.hough_line_peaks()를 이용하여 투표수가 높은 주요 직선을 추출하였다. num_peaks=10으로 최대 10개의 직선을 선택하고, peak_threshold=0.2*hspace.max()를 적용하여 최대 투표수의 20% 이상인 후보를 선택 기준으로 설정하였다.
최종적으로 검출된 직선의 ρ와 θ를 이용하여 plt.axline()으로 원본 영상 위에 빨간색 직선을 표시하였다.

### 5.2. p3.jpg 및 p4.jpg — 주요 직선 검출 결과

<img width="940" height="269" alt="image" src="https://github.com/user-attachments/assets/c4bc1d79-7354-4a39-97ba-890f68d03286" />
<img width="940" height="270" alt="image" src="https://github.com/user-attachments/assets/e9037376-ec3c-4d35-83ce-8a23d4a7b30a" />

p3.jpg에서는 건물의 지붕, 외벽, 창문 등의 경계가 Canny Edge Detector를 통해 검출되었다. Hough Transform을 적용한 결과, 지붕의 기울어진 평행선과 건물의 수직 경계 등 주요 직선 구조가 추출되었다. 특히 지붕을 따라 유사한 기울기를 가진 여러 직선이 검출되어 실제 건물의 구조와 비교적 잘 일치하였다.
p4.jpg에서는 사람, 카메라, 삼각대 및 배경의 경계가 검출되었다. Hough Transform 결과에서는 삼각대의 다리와 중앙 지지대, 배경의 수평선 등이 주요 직선으로 나타났다. 일부 직선은 사람이나 카메라 주변을 가로질렀는데, 이는 Hough Transform이 물체의 의미가 아닌 Edge 픽셀의 투표수를 기준으로 직선을 선택하기 때문이다.
두 영상 모두 주요 직선 구조를 성공적으로 검출하였으며, num_peaks=10을 유지하여 결과를 비교하였다.


### 5.3. p5.jpg 및 p6.jpg — 경추 영상 비교

<img width="940" height="273" alt="image" src="https://github.com/user-attachments/assets/65f5e2bd-37a7-403f-af44-3fe91aa6b55a" />
<img width="940" height="271" alt="image" src="https://github.com/user-attachments/assets/860e915d-6497-4704-9f6e-e3f67f95be7c" />

p5.jpg와 p6.jpg의 경추 X-ray에 동일한 Canny 및 Hough Transform을 적용하였다.
두 영상 모두에서 경추뼈의 윤곽과 주변 조직의 경계가 검출되었으며, Hough Accumulator에서 투표수가 높은 주요 직선들이 추출되었다.
원본 영상을 시각적으로 비교한 결과, p5.jpg에서는 경추의 완만한 C자형 곡률이 상대적으로 관찰되는 반면, p6.jpg에서는 경추가 보다 직선화된 형태로 나타났다.
따라서 문제에서 제시한 정상적인 C자형 경추 곡률을 기준으로 시각적으로 비교하면, p6.jpg가 p5.jpg보다 거북목에 가까운 영상으로 판단된다.

### 5.4. 결과 및 고찰

Canny Edge Detection과 Hough Transform을 연속적으로 적용하여 영상의 주요 직선 경계를 추출할 수 있었다.
p3와 p4에서는 건물과 삼각대처럼 직선 구조가 뚜렷한 물체의 경계를 효과적으로 검출하였다. 반면 p5와 p6에서는 경추의 복잡한 곡선 구조를 여러 직선 후보로 표현하기 때문에, 검출 결과만으로 경추의 실제 곡률을 정확하게 판단하기에는 한계가 있었다. Hough Transform은 주요 직선 구조를 검출하는 데 효과적이지만, 곡선 형태의 해부학적 구조를 평가할 때는 추가적인 분석이 필요하다.
