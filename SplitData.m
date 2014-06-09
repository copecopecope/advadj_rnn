function [goldDist, map] = SplitData(filename, minPairs, outTest, outTrain)

filename = ['data-hold/', filename];
fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

A = containers.Map;

maxLine = length(C{1});

disp('Generating map...')

for i=1:maxLine
  if mod(i,1000) == 0
     fprintf('.');
  end
  if ~isempty(C{1}{i})
    splitLine = textscan(C{1}{i}, '%s', 'delimiter', ',');
    splitLine = splitLine{1};

    rating = splitLine{1};
    rating = str2num(rating);
    adv = splitLine{2};
    adj = splitLine{3};
    advadj = strcat(adv,',',adj);
    
    if ~isKey(A,advadj)
       A(advadj) = zeros(1,10);
    end
    dist = A(advadj);
    dist(rating) = dist(rating) + 1;
    A(advadj) = dist;
  end
end
fprintf('\n');

allKeys = keys(A);
lenKeys = length(allKeys);

fprintf('Filtering out pairs with less than %d instances...\n', minPairs);
for i=1:lenKeys
  key = allKeys(i);
  key = key{1};
  numPairs = sum(A(key));
  if numPairs < minPairs
     remove(A,key);
  end
end

allKeys = keys(A);
lenKeys = length(allKeys);
fprintf('Number of remaining pairs: %d\n', lenKeys);

goldDist = zeros(lenKeys,10);
map = containers.Map;
totalPairs = 0;
for i=1:lenKeys
  key = allKeys(i);
  key = key{1};
  map(key) = i;
  goldDist(i,:) = A(key);
  numPairs = sum(A(key));
  totalPairs = totalPairs + numPairs;
  fprintf('%s %d\n',key,numPairs);
end
fprintf('Total pairs: %d\n', totalPairs);
    
    
    
    
