function GenerateWordList(inFile, outFile)

fid = fopen(inFile);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

maxLine = length(C{1});
wordList = [];

for line = 1:maxLine
    splitLine = textscan(C{1}{line}, '%s', 'delimiter', ',');
    splitLine = splitLine{1};
    words = [splitLine(2) splitLine(3)];
    wordList = [wordList words];
end

wordList = unique(wordList);

fid = fopen(outFile,'w');
for i=1:length(wordList)
  fprintf(fid,'%s\n',char(wordList(i)));
end
fclose(fid);
    
