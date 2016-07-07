ImageID = fopen('/Users/bengiugliano/Downloads/train-images.idx3-ubyte')
%Read imge file into 1 long column vector, 'LongImgVector'
LongImgVector= fread(ImageID);
%Open Label file
labelID = fopen('/Users/bengiugliano/Downloads/train-labels-2.idx1-ubyte');
%Read Label file into column vector, 'LabelVector'
LabelVector = fread(labelID);
%Find all places in the label array where there is a 0 or 1, and place in
%an array.
SVMModel = zeros(9,1);
for count = 0: 1: 8
   for num = count+1: 9
       numberSmall = find(LabelVector==count);
       numberSmallLabels = zeros(length(numberSmall),1);
       for k = 1:length(numberSmall)
         numberSmallLabels(k,1) = count;
       end
       numberLarge = find(LabelVector==num);
       numberLargeLabels = zeros(length(numberLarge),1);
       for k = 1:length(numberLarge)
         numberLargeLabels(k,1) = num;
       end
       numberSmall = numberSmall - 8;
       numberLarge = numberLarge - 8;
       % get small and large number ^
       smallArray = zeros(784, length(numberSmall));
       for i=1: length(numberSmall)
           if numberSmall(i)>0
               start=16+((numberSmall(i)-1)*784);
               stop=16+(numberSmall(i)*784);
               for t=start:(stop-1)
                   smallArray((t+1-start),i)=LongImgVector(t);
               end
           end
       end
       smallArray = transpose(smallArray);
       start(:) =[];
       stop(:)=[];
       %%%%%%%%%%% large num images
       largeArray = zeros(784, length(numberLarge));
       for i=1: length(numberLarge)
           if numberLarge(i)>0
               start=16+((numberLarge(i)-1)*784);
               stop=16+(numberLarge(i)*784);
               for t=start:(stop-1)
                   largeArray((t+1-start),i)=LongImgVector(t);
               end
           end
       end
       largeArray = transpose(largeArray);
       start(:) =[];
       stop(:)=[];
       
       %%%%%%%%%% SVM stuff
       combinedImages = [smallArray;largeArray];
       combinedLabels = [numberSmall;numberLarge];
       %combinedLabelsGood = transpose(combinedLabels);
       SVMModel((count+1),1) = fitcsvm(combinedImages,combinedLabels,'KernelFunction','linear','Standardize',true);
       combinedImages(:) =[];
       combinedLabels(:) =[];
   end
end
