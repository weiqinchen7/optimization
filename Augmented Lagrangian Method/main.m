load CTReconPhantom
a=zeros(589824,1);
v=zeros(30720,1);
u=100;
rou=u/2;
tau=0.0000001;
for outer_i=1:700
    for inner_i=1:50
        a_l=reshape(a,[256,256,9]);
        wt_a=iswt2(a_l,1,1); % 256*256
        wt_a_clmn=reshape(wt_a,[65536,1]);
        aawtap=A'*(A*wt_a_clmn-p); % 65536*1
        aawtap_n=reshape(aawtap,[256,256]);
        waawap=swt2(aawtap_n,1,1); % 256*256*9
        waawap_clmn=reshape(waawap,[589824,1]);
        Av=A'*v; %65536*1
        Av1=reshape(Av,[256,256]);
        wav= swt2(Av1,1,1); % 256*256*9
        wav_clmn=reshape(wav,[589824,1]);
        shrink_element=a_l-tau*(u*waawap-wav); % 256*256*9
        
        a_mid=shrink(shrink_element,tau); % 1*589824
        a=reshape(a_mid,[256,256,9]);
    end
    
    wt_a_2=iswt2(a,1,1); % 256*256
    wt_a_2_clmn=reshape(wt_a_2,[65536,1]);
    v=v+rou*(p-A*wt_a_2_clmn); % 30720*1
end

result_I = iswt2(a,1,1);

figure;
imshow(result_I,[]);