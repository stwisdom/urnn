% Reads original TIMIT files, downsamples them to desired sampling frequency,
% then writes the result to .wav files in specified directory 'outdir'.

clear variables;
addpath('rdir');

% change this to your TIMIT path:
timitdir='../TIMIT';

fs=8e3;
outdir='../TIMIT_8khz';
%fs=16e3;
%outdir='../TIMIT_16khz';
num_segments=1;

mkdir(outdir);


files=rdir(timitdir + '/**/**/**/*.WAV');
files={files(:).name};

for ifile=1:length(files)
    [x,fs_orig]=audioread(files{ifile});
    x=x(:,1);
    x=resample(x,fs,fs_orig);
    xlen=length(x);
    seglen=floor(xlen/num_segments);
    idx=1;
    for iseg=1:num_segments
        xseg=x(idx:idx+seglen-1);
        idx=idx+seglen;
        [filepath,filename]=fileparts(files{ifile});
        if num_segments>1
            filename=[filename, sprintf('_seg%d',iseg)];
        end
        path_split=strsplit(filepath,'TIMIT/');
        outdir_cur=fullfile(outdir,path_split{2});
        if ~exist(outdir_cur,'dir')
            mkdir(outdir_cur);
        end
        wavwrite(xseg,fs,fullfile(outdir_cur,[filename '.wav']));
    end
end

