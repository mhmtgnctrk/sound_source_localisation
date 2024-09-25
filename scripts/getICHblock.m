function [ fICH, nout, x ] = getICHblock( fICH )
%GETICHBLOCK get a block of data from an ich file

% [ fICH, n ] = getICHblock( fICH ) reads just the header from the
%   file and advances the file pointer.  This is useful for
%   scanning the file. n is the number of the block just read.
%
% [ fICH, n, x ] = getICHblock( fICH ) reads a block of samples and puts
%   them into Lx16 array x.  If the 'skip' option was set when the
%   file was opened, x is filled and the file structure updated
%   to the currend sequence.  Otherwise, if the "next" packet in the
%   sequence is missing, x is left empty, the file pointer is NOT
%   advanced, and n is assigned the next number in the sequence.
%
% In either case, n is adjusted to account for the 16 bit rollover,
% but set to -1 if the end of the file is reached, or another error
% is encountered (packet header length wrong).

[ packetHead, ~ ] = fread( fICH.F, 2, 'uint16' );
if feof( fICH.F )
    nout = -1;
    x = [];
    return
end

if packetHead(1)~=fICH.samplespp*32
    warning( 'getICHblock:packetheader', ...
        'Packet %d (wanted %d) header has bad payload length (%d)!', ...
        packetHead(2), fICH.nextBlock, packetHead(1) );
end

% handle sequence error
if packetHead(2)~=mod(fICH.nextBlock,2^16)
    fprintf( 'packet header number %d, internal counter %d (%d): %8.3fs\n', ...
        packetHead(2), fICH.nextBlock, mod(fICH.nextBlock,2^16), fICH.nextBlock*.004 );
    if fICH.skip==0
        nout = fICH.nextBlock;
        fICH.nextBlock = fICH.nextBlock+1;
        fseek( fICH.F, -4, 0 );  % rewind stream to beginning of packet
        x = zeros( fICH.samplespp, 16 );
        return;
    end
end

nout = fICH.nextBlock;

if nargout>2
    packetSize = packetHead(1)/2;
    [ packetData, l ] = fread( fICH.F, packetSize, 'int16' );
    if l~=packetSize
        warning( 'getICHblock:shortBlock', 'Read less than expected from packet %d.', ...
            fICH.nextBlock );
        nout = -1;
        x = [];
        return;
    end
    
    % each block of data is effecively divided into two subblocks, one for
    % each ~2ms of data. Within this subblock the samples are arranged by
    % channel, so 2ms of channel 1 data, then 2ms of channel 2 data, etc.
    x = [ reshape(packetData(1:(packetSize/2)), packetSize/32, 16); ...
        reshape(packetData((packetSize/2+1):packetSize), packetSize/32, 16) ];
else
    % skip ahead to next block
    fseek( fICH.F, packetHead(1), 0 );
end

fICH.nextBlock = fICH.nextBlock+1;

end

