function c = AT(y, M, N)
    c = fft([y  zeros(1, N-M)]);
end