function ich2wav( filename, N, channels )

switch nargin;
    case 1
        N = [ 0 Inf ];
        channels = 1:16;
    case 2
        channels = 1:16;
    case 3
        % do nothing
    otherwise
        error( 'Invalid number of arguments.' );
end

f = openICH( [ 'ICH/' filename '.ich' ] );

% if file does not have enough blocks, reduce the length
if f.nBlocks<(N(1)+N(2))
    N(2) = f.nBlocks-N(1);
end

x = zeros(N(2)*f.samplespp, length(channels), 'int16');

ProgressBar( 0, 0.01, 'Converting...' );

% skipping beginning of file
c = 0;
while c<N(1)
    [ f, c ] = getICHblock( f );
    if c==-1
        break
    end
    ProgressBar( c/(N(1)+N(2)) );
end

while c<(N(1)+N(2))
    [ f, c, newdata ] = getICHblock( f );
    if c==-1
        break
    end
    
    x(((c-N(1))*f.samplespp)+(1:f.samplespp),:) = newdata(:,channels);
    
    ProgressBar( c/(N(1)+N(2)) );
end
ProgressBar();

ProgressBar( 0, 0.01, 'Writing...' );
mkdir( '48k', filename );
for n=1:length(channels)
    wavwrite( x(:,n), f.fs, ...
        sprintf( '48k/%s/ch%02d.wav', filename, channels(n) ));
    ProgressBar( n/(2*length(channels)) );
end

x16k = resample( double(x)/(2^15), 16000, f.fs );
mkdir( '16k', filename );
for n=1:length(channels)
    wavwrite( x16k(:,n), 16000, ...
        sprintf( '16k/%s/ch%02d.wav', filename, channels(n) ));
    ProgressBar( (length(channels)+n)/(2*length(channels)) );
end
ProgressBar();

fprintf( 'wav files written.\n' );


