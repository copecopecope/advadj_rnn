% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ wordMap, relationMap, relations, data ] = ...
    LoadTrainingData(filename)
% Load word-word pair data for pretraining and to generate a word map.

% For some experiments, this is only used to initialize the words and
% relations, and the data itself is not used.


% Load the file
fid = fopen(filename);
disp(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Initialize the data array
rawData = repmat(struct('relation', 0, 'leftText', '', 'rightText', ''), ...
    length(C{1}), 1);
wordList = cell(length(C{1}), 1);

% Establish (manually specified) relations
relations = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'};
relationMap = containers.Map(relations,1:length(relations));

% Parse the file
itemNo = 1;
wordNo = 1;
maxLine = length(C{1});
%maxLine = 10; % Uncomment to truncate data for testing.

for line = 1:maxLine;
    if ~isempty(C{1}{line}) 
        splitLine = textscan(C{1}{line}, '%s', 'delimiter', '\t');

        splitLine = splitLine{1};
                %disp(splitLine);
        %if ~(length(splitLine{1}) ~= 1 || splitLine{1} == '%')
            % Skip lines that are blank or have a multicharacter first chunk
            %rawData(itemNo).relation = relationMap(splitLine{1});
            rawData(itemNo).leftText = splitLine{1};
            %disp('+');
            %disp(rawData(itemNo).leftText);
            %rawData(itemNo).rightText = splitLine{3};

            % Add to wordList
            lWords = textscan(splitLine{1}, '%s', 'delimiter', ' ');
            rWords = textscan(splitLine{1}, '%s', 'delimiter', ' ');
            words = unique([lWords{1}; rWords{1}]);
            wordList(wordNo:wordNo + (length(words) - 1)) = cellstr(words);
            wordNo = wordNo + length(words);

            itemNo = itemNo + 1;
        %end
    end
end

rawData = rawData(1:itemNo - 1);

% Compile vocabulary
wordList = wordList(1:wordNo - 1);
vocabulary = unique(wordList);

% Remove syntactic symbols from vocabulary
vocabulary = setdiff(vocabulary, {'(', ')'});


% Build word map
wordMap = containers.Map(vocabulary,1:length(vocabulary));

% This isn't optimized for cases where we don't need the data itself, this 
% is just a shortcut:
if nargout > 3
    % Build the dataset
    data = repmat(struct('relation', 0, 'leftTree', Tree(), 'rightTree', Tree()), ...
        length(rawData), 1);

    % Build Trees
    for dataInd = 1:length(rawData)
        data(dataInd).leftTree = Tree.makeTree(rawData(dataInd).leftText, wordMap);
        data(dataInd).rightTree = Tree.makeTree(rawData(dataInd).rightText, wordMap);
        data(dataInd).relation = rawData(dataInd).relation;
    end
    % data = [data; Symmetrize(data)];
end

disp('Done Loading Training');

end

