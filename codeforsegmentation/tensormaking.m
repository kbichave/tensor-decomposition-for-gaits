function Datachunk=tensormaking(Data,crossingsofinterest,avglength,lengthchunk)
Datachunk=zeros(length(crossingsofinterest)-1,3,avglength);
for datachunkcounter=1:length(crossingsofinterest)-1
 if(lengthchunk(datachunkcounter)<avglength)
     t1=mSTRETCH(Data{crossingsofinterest(datachunkcounter):crossingsofinterest(datachunkcounter+1)-1,4},avglength);
     t2=mSTRETCH(Data{crossingsofinterest(datachunkcounter):crossingsofinterest(datachunkcounter+1)-1,5},avglength);
     t3=mSTRETCH(Data{crossingsofinterest(datachunkcounter):crossingsofinterest(datachunkcounter+1)-1,6},avglength);
     Datachunk(datachunkcounter,:,:)=transpose([t1 t2 t3]);
 end
 if(lengthchunk(datachunkcounter)>avglength)
     t1=mPRESS(Data{crossingsofinterest(datachunkcounter):crossingsofinterest(datachunkcounter+1)-1,4},avglength);
     t2=mPRESS(Data{crossingsofinterest(datachunkcounter):crossingsofinterest(datachunkcounter+1)-1,5},avglength);
     t3=mPRESS(Data{crossingsofinterest(datachunkcounter):crossingsofinterest(datachunkcounter+1)-1,6},avglength);
     Datachunk(datachunkcounter,:,:)=transpose([t1 t2 t3]); 
 end
  if(lengthchunk(datachunkcounter)==avglength)
     Datachunk(datachunkcounter,:,:)=transpose(Data{crossingsofinterest(datachunkcounter):crossingsofinterest(datachunkcounter+1),4:6})
 end
 
end


end