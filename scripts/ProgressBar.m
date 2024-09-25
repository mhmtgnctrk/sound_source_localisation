function ProgressBar (Frac, Delta, Title)
% Update a progress bar window. The progress bar shows the progress of a
% computation with an annotation of progress in percent. This routine is a
% wrapper around the Matlab function waitbar. Alongside the bar, it prints
% the progress in percent. The granularity between updates to the bar can
% be set. For instance if the granularity is set to 5%, the bar will only
% be updated when the progress crosses a 5% boundary. This reduces the
% computational load for situations in which there are many small updates
% to the progress.
%
% There are three different calls to this routine.
% 1. Two or more arguments - initialize the routine
% 2. One argument - update the progress
% 3. No argument - close the progress bar
%
% Initialization:
% h = ProgressBar(Frac, Delta, Title)
% h = ProgressBar(Frac, Delta)          % Default Title
% h = ProgressBar(Frac, Title)          % Default Delta
% - Frac is the starting progress fraction (0 to 1).
% - Delta specifies the increment (0 to 1) between updates to the progress
%   bar window. The progress bar window is updated every time the progress
%   crosses a multiple of Delta. For instance if Delta is 0.05, the
%   progress is updated when the progress crosses 5%, 10%, ... . If Delta
%   is small enough, the progress bar window is updated for every update
%   call. The default value for Delta is 0.01, which implies at most 100
%   updates to the visible progress bar (assuming the progress values are
%   monotonically increasing).
% - Title appears as the title of the progress bar window. The default is
%   no title.
% - h is the handle to the waitbar figure. This can be used for additional
%   customization (colours, etc.) of the progress bar.
%
% Update:
% ProgressBar(Frac)
% - Frac is the fraction (0 to 1) of the progress. If the progress
%   crosses a multiple of Delta, the progress bar window is updated.
%
% Close:
% ProgressBar()
% Close the progress bar.

% Example:
%  ProgressBar(0, 0.02, 'Progress ...');
%  for (i = 1:M)
%    ProgressBar((i-1)/(M-1)); % Update bar if appropriate
%    ...
%  end
%  ProgressBar();        % Close the progress bar

% $Id: ProgressBar.m,v 1.1 2010/02/24 17:56:35 pkabal Exp $
% by Peter Kabal, with modifications by Joachim Thiemann

persistent nsFrac h sDelta
BarColor = [0.5 0.5 1];    % Bar colour

if (nargin > 1)

% Initialization - close waitbar if it is open
  if (ishandle(h))
    close(h);
    h = [];
  end

% Result Delta / Title
  if (nargin == 2)
    if (ischar(Delta))
      Title = Delta;
      Delta = 0.01;       % Default Delta value
    else
      Title = '';         % Default Title
    end
  end
  sDelta = Delta;

  h = InitWaitBar(Frac, Title, BarColor);

  % Set the initial nsFrac value
  if (sDelta > 0)
    nsFrac = floor(Frac / sDelta);
  else
    nsFrac = Inf;
  end

elseif (nargin == 1)

% Update the progress
  if (sDelta > 0)
    nFrac = floor(Frac / sDelta);
    if (nFrac ~= nsFrac)
      UpdateWaitBar(h, Frac);
      nsFrac = nFrac;
    end
  else
    UpdateWaitBar(h, Frac);
  end

else

% Close the waitbar figure
  if (ishandle(h))
    close(h);
  end
  h = [];
end

return

% ----- -----
function h = InitWaitBar (Frac, Title, BarColor)

Text = sprintf('%2.0f%% ...', 100 * Frac);
% if-statement code to check for display by Amro found on
% http://stackoverflow.com/questions/6754430/determine-if-matlab-has-a-display-available
if usejava('jvm') && ~feature('ShowFigureWindows')
  h = -1;
else
  h = waitbar(Frac, Text, 'Name', Title);
  ph = findobj(h, 'type', 'patch'); 
  set(ph, 'FaceColor', BarColor, 'EdgeColor', BarColor); 
end

return

% ----- -----
function UpdateWaitBar (h, Frac)
% waitbar will clamp Frac to between 0 and 1

if h ~= -1
  Text = sprintf('%2.0f%% ...', 100 * Frac);
  waitbar(Frac, h, Text);
else
  fprintf( ' %3.0f%%   \r', 100*Frac );
end

return
