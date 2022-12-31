function N = RandomNoise(sz, weights)

N = zeros(sz);
for i = 1:length(weights)
  szi = round(sz / 2^(i - 1));
  %Ni = weights(i) * RandCentered(szi);
  Ni = weights(i) * randn(szi);
  N = N + imresize(Ni, sz, 'bicubic');
end

N = N - mean(N(:));
N = N / std(N(:));
