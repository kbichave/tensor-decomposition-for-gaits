function [crossingsofinterest lengthchunk ]=chopi(Data, i,ilimit,l1)
countermin=0;
countermax=0;
imax=i;
% while (imax<=ilimit)
%     if smooth(Data{imax,6})<l2
%         countermin=countermin+1;
%         minimums(countermin)=imax;
%         imax=imax+40;
%     end
% 
%     imax=imax+1;
% end

imax=i;
while (imax<=ilimit)
    if smooth(Data{imax,4})>l1 
        countermax=countermax+1;
        maximums(countermax)=imax;
        disp(imax);
        imax=imax+50;
        
        
    end

    imax=imax+1;
end
maximums=transpose(maximums);
% minimums=transpose(minimums);
crossingsofinterest=zeros(length(maximums),1);
for zerocrossing=1:length(maximums)
%     minpoint=minimums(zerocrossing);
    maxpoints=maximums(zerocrossing);
    for pointfinder=maxpoints:maxpoints+16
        if(Data{pointfinder+1,4}>Data{pointfinder,4})
            crossingsofinterest(zerocrossing)=pointfinder;
            break;
        end
    end
end
lengthchunk=zeros(length(crossingsofinterest)-1,1)
for i=1:length(crossingsofinterest)-1
    lengthchunk(i)=(crossingsofinterest(i+1)-crossingsofinterest(i))+1;
end

end