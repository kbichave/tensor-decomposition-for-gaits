function [crossingsofinterest lengthchunk ]=chopi(Data, i,ilimit,l1,l2)
countermin=0;
countermax=0;
imax=i
while (imax<=ilimit)
    if smooth(Data{imax,6})<l2
        countermin=countermin+1;
        minimums(countermin)=imax;
        imax=imax+40;
    end

    imax=imax+1;
end

imax=i
while (imax<=ilimit)
    if smooth(Data{imax,6})>l1 
        countermax=countermax+1;
        maximums(countermax)=imax;
        imax=imax+40;
    end

    imax=imax+1;
end
maximums=transpose(maximums);
minimums=transpose(minimums);
crossingsofinterest=zeros(length(minimums),1);
for zerocrossing=1:length(minimums)
    minpoint=minimums(zerocrossing);
    maxpoint=maximums(zerocrossing);
    for pointfinder=minpoint-20:minpoint
        if(Data{pointfinder,6}>=0 & Data{pointfinder+1,6}<0)
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