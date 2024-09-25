function [ fICH ] = openICH( filename, opt )
%OPENICH open the ich file and read header parameters

sratetable = [ 48000 32000 16000 8000 44100 22050 11025 ];
psizetable = [   192   128    64   32   192    96    48 ];

fICH.F = fopen( filename, 'rb' );
if fICH.F==-1
    error( 'Could not open file %s.', filename );
end

fICH.skip = 0;
if nargin>1
    if strcmp(opt,'skip')
        fICH.skip = 1;
    end
end

% read header
head = fread( fICH.F, 4, 'uint32' );
if size(head)<4
    fclose( fICH.F );
    error( 'Could not read header.' );
end    

% evaluate header: file format
fICH.isICH = 0;
if head(1) == 2^28
    fICH.isICH = 1;
elseif head(1) == 2^29
    % file is OCH; leave isICH = 0.
else
    fclose( fICH.F );
    error( 'Format not recognized (first longword of header)' );
end

% evaluate header: sampling rate
if head(2)>6
    fclose( fICH.F );
    error( 'Sampling rate of file not recognised.' );
end

fICH.fs = sratetable(head(2)+1);
fICH.samplespp = psizetable(head(2)+1);  % samples per channel per packet
fICH.packetSize = 32*fICH.samplespp+4;   % length in bytes of packet

% evaluate header: number of channels
if head(3)~=0 % only '0' (=16channels) is allowed
    fclose( fICH.F );
    error( 'Number of channels other than 16 not supported.' );
end

fICH.nextBlock = 0;

% find out how big the file is
fseek(fICH.F,0,'eof'); % go to end
filesize = ftell(fICH.F);
fseek(fICH.F,16,'bof'); % go back to first block

% and convert to blocks
fICH.nBlocks = floor((filesize-16)/fICH.packetSize);
if (fICH.nBlocks*fICH.packetSize)<(filesize-16)
    warning( 'openICH:consistency', 'File size is not consistent.' );
end

end
