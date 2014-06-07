% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ data ] = LoadConstitData(filename, wordMap, relationMap, hyperParams)
% Load one file of constituent-pair data.

% Append data-4/ if we don't have a full path:
if isempty(strfind(filename, '/'))
    if strfind(filename, 'quant_')
        filename = ['grammars/data/', filename];
    else   
        filename = ['data-4/', filename];
    end
end
fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Load the file

% Initialize the data array
rawData = repmat(struct('relation', 0, 'leftText', '', 'rightText', ''), ...
    length(C{1}), 1);

% Parse the file
itemNo = 1;
maxLine = length(C{1});
% maxLine = 25;

for line = 1:maxLine
    if ~isempty(C{1}{line}) 
        splitLine = textscan(C{1}{line}, '%s', 'delimiter', ',');
        splitLine = splitLine{1};

        
        if ~(length(splitLine{1}) ~= 1 || splitLine{1} == '%')
            % Skip lines that are blank or have a multicharacter first chunk
            rawData(itemNo).relation = relationMap(splitLine{1});
%                     fprintf(splitLine{1});
%                     fprintf(' : ');
            rawData(itemNo).leftText = splitLine{2};
%                     fprintf(splitLine{2});
%                     fprintf(' :: ');
            rawData(itemNo).rightText = splitLine{3};
%                     fprintf(splitLine{3});
%                     fprintf(' :: ');

            itemNo = itemNo + 1;
        end
    end
end

disp('Done Reading in File');

rawData = rawData(1:itemNo - 1);

% Build the dataset
data = repmat(struct('relation', 0, 'leftTree', Tree(), 'rightTree', Tree()), ...
    length(rawData), 1);

% Build Trees
for dataInd = 1:length(rawData)
    data(dataInd).leftTree = Tree.makeTree(rawData(dataInd).leftText, wordMap);
    data(dataInd).rightTree = Tree.makeTree(rawData(dataInd).rightText, wordMap);
    data(dataInd).relation = rawData(dataInd).relation;
end

disp('Done Making Trees');

end

