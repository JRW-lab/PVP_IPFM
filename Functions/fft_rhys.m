function fwindows = fft_rhys(twindows,fs,f0,Twin,type)

if size(twindows,2) == fs*Twin
    twindows = twindows.';
end

switch type
    case "mag"
        fwindows = abs(fft(twindows)) / fs;
        fwindows = fwindows(1:(f0*Twin),:).';
    case "all"
        fwindows = fft(twindows) / fs;
        fwindows = fwindows(1:(f0*Twin),:).';
        fwindows = [real(fwindows) imag(fwindows(:,2:end))];
end