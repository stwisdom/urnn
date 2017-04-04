function score_timitpred(exp_list,audio_folder)

    addpath('matlab/evaluation/voicebox') %add voicebox path                                
    addpath('matlab/evaluation/obj_evaluation') %add Loizou objective evaluation path       
    addpath('matlab/evaluation/stoi')   %add STOI path 

    if ~exist('audio_folder','var')
        audio_folder = 'audio_output';
    end
    fprintf('Removing all wav files from audio folder %s\n',audio_folder);
    unix(sprintf('rm %s/*.wav',audio_folder),'-echo');

    fid_exp_list = fopen(exp_list,'r');
    exp_cur = fgetl(fid_exp_list);
    while ischar(exp_cur)

        fprintf('Scoring experiment %s\n\n',exp_cur);
        cmd=sprintf('python2.7 -u recon_timitpred.py -s %s -o %s',exp_cur,audio_folder);
        fprintf('Running command %s\n\n',cmd);
        unix(cmd,'-echo');

        files=dir([audio_folder,'/est*.wav']);
        files={files.name}; 
        nfiles=length(files);
        est=fullfile(audio_folder,files{1});
        ref=strrep(est,'est','ref');
        [S1,labels]=compute_scores(est,ref);
        S=zeros(nfiles,length(S1));
        S(1,:)=S1;
        tic;
        for ifile=2:nfiles
            if mod(ifile,floor(nfiles/10))==0
                fprintf('Scored %d files of %d total\n',ifile,nfiles);
                toc;
                tic;
            end
            est=fullfile(audio_folder,files{ifile});
            ref=strrep(est,'est','ref');
            S(ifile,:)=compute_scores(est,ref);
        end
        save([exp_cur,'_scores.mat'],'S','labels');
        for iscore=1:length(S1)
            fprintf('Mean %s = %.2f\n',labels{iscore},mean(S(:,iscore)));
        end

        exp_cur=fgetl(fid_exp_list);

    end

    fclose(fid_exp_list);

end

function [S,labels]=compute_scores(est,ref)
    
    % read audio
    xest=wavread(est);
    xref=wavread(ref);

    len_est=length(xest);
    len_ref=length(xref);
    len_min=min(len_est,len_ref);
    xest=xest(1:len_min);
    xref=xref(1:len_min);

    % segmental SNR
    [loc,glo]=snrseg(xest,xref,8000.0);

    % PESQ
    pesq_mos=pesq(ref,est);

    % STOI
    stoi_score=stoi(xref,xest,8000.0);

    S=[loc, glo, pesq_mos, stoi_score];
    if nargout>1
        labels={'SegSNR local','SegSNR global','PESQ','STOI'};
    end

end

