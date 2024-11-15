
[VAE]

0. 사전지식
- https://medium.com/@flatorien/ai-%EC%8B%9C%EB%8C%80-%ED%86%B5%EA%B3%84-%EC%82%AC%EC%9A%A9%EC%9E%90%EC%9D%98-%EC%97%AD%ED%95%A0-%EC%9E%91%EC%84%B1-1fb987c5e94f
- 
- https://medium.com/@flatorien/variational-auto-encoder-%EC%99%80-latent-layer%EC%9D%98-%EB%B6%84%ED%8F%AC-%EA%B3%A0%EC%A0%95-d4e2ac5db4d8
- 
- https://medium.com/@flatorien/vae%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-anomaly-detection-c630251f4f28

  
3. Likelihood (우도)란 무엇인가?

https://youtu.be/XepXtl9YKwc?si=chiDUt46Fl3Z84a7
https://jjangjjong.tistory.com/41
https://data-scientist-brian-kim.tistory.com/91
https://lcyking.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-VAEVariational-Auto-Encoder

2. VAE

https://medium.com/@hugmanskj/hands-on-understanding-and-implementing-variational-autoencoders-1013b658d216

https://medium.com/@hugmanskj/autoencoder-%EC%99%80-variational-autoencoder%EC%9D%98-%EC%A7%81%EA%B4%80%EC%A0%81%EC%9D%B8-%EC%9D%B4%ED%95%B4-171b3968f20b

https://hugrypiggykim.com/2018/09/07/variational-autoencoder%EC%99%80-elboevidence-lower-bound/

https://ijdykeman.github.io/ml/2016/12/21/cvae.html

3.  변분 추론(Variational Inference)

https://medium.com/@david.daeschler/the-amazing-vae-part-2-06927b916363

https://medium.com/@rukshanpramoditha/a-comprehensive-guide-to-variational-autoencoders-vaes-708be64e3304

https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73

- 실제 데이터 분포 p(x)에 가장 가까운 데이터를 생성하는 모델의 매개변수를 학습하고 싶음
- 베이즈 정리를 사용하여 posterior probability p(Θ|X)를 계산해야 함
- 하지만, 수학적 난해성 때문에 p(Θ|X)를 직접 계산할 수 없음
- 따라서, 복잡한 확률 분포로 근사값을 찾음
- 이를 위해서, 잠재 공간의 분포가 정규 분포(e.g Gaussian, Bernulli)를 따를 것이라고 가정하고 잠재 변수 𝑧를 샘플링
(잠재 공간을 점 추정하는 것이 아니라 공간을 샘플링,  Z ≈ N(µx, σx))
- 직접 샘플링하면 미분이 불가능해지므로, 재파라미터화 트릭(Z = μ + σϵ)형태로 변환
- ϵ은 표준 정규 분포 N(0,1)로부터 샘플링된 노이즈
- 샘플링된 z는 디코더로 전달되어 원본 데이터 x와 유사한 데이터를 생성하는 데 사용
- VAE의 목적 함수는 Evidence Lower BOund(ELBO)을 최대화
- ELBO는 재구성 오차와 KL 발산의 합으로 구성
- 재구성 오차는 입력 데이터와 재구성된 데이터 간의 차이를 최소화
- KL 발산(Kullback-Leibler Divergence)은 두 확률 분포 간의 차이를 측정하는 지표
- VAE에서 KL 발산은 <인코더가 학습한 잠재 변수의 분포>와 모델이 원하는 목표 분포(일반적으로 표준 정규 분포) N(0,1)) 사이의 차이를 최소화하는 데 사용
