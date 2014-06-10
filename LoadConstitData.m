% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ data ] = LoadConstitData(filename, wordMap, relationMap, hyperParams)
% Load one file of constituent-pair data.

disp(filename)

filename = ['data/' filename];
fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Load the file

% Initialize the data array
rawData = repmat(struct('relDist', zeros(10,1), 'leftText', '', 'rightText', ''), ...
    length(C{1}), 1);

% Parse the file
itemNo = 1;
maxLine = length(C{1});
% maxLine = 25;

disp(maxLine)

for line = 1:maxLine
    if ~isempty(C{1}{line}) 
        splitLine = textscan(C{1}{line}, '%s', 'delimiter', ',');
        splitLine = splitLine{1};

        for i = 1:10
            rawData(itemNo).relDist(i) = str2num(char(splitLine{i+2})) + 1; % 1 for regularization
        end
        rawData(itemNo).relDist = rawData(itemNo).relDist / sum(rawData(itemNo).relDist); % normalize
        rawData(itemNo).leftText = splitLine{1};
        rawData(itemNo).rightText = splitLine{2};
        itemNo = itemNo + 1;
    end
end

disp('Done Reading in File');

rawData = rawData(1:itemNo - 1);


% Build the dataset
data = repmat(struct('relDist', 0, 'leftTree', Tree(), 'rightTree', Tree()), ...
    length(rawData), 1);

% Build Trees
for dataInd = 1:length(rawData)
    data(dataInd).leftTree = Tree.makeTree(rawData(dataInd).leftText, wordMap);
    data(dataInd).rightTree = Tree.makeTree(rawData(dataInd).rightText, wordMap);
    data(dataInd).relDist = rawData(dataInd).relDist;
end

disp('Done Making Trees');

end

