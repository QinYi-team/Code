function y = A(c, M, N)
    v = N * ifft(c);
    y = v(1:M);
end