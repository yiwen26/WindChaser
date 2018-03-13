%Create the window for user interaction
prompt = {'Enter testing dataset location:','Enter forecasting interval:'};
dlg_title = 'Input Dialog';
num_lines = 1;
defaultans = {'20','hsv'};
answer = inputdlg(prompt,dlg_title,num_lines,defaultans);


%Read in data from specific filenames. As an example here, we load data
%from current saved files.
orig=csvread('orig test.csv');
pred=csvread('pred test.csv');
pred=pred*16;
orig=orig*16;

%Create the real-time data read-in and plot the real-time results. Use some
%hyperparameters in the future.
fig1 = figure;
pos_fig1 = [0 0 1920 1080];
set(fig1,'Position',pos_fig1)
xlabel('Time of a day(hour)')
ylabel('Wind Power (MWh)')
h = animatedline('color','r');
h2=animatedline('color','b');
axis([0 24 0 16])
x = linspace(0,24,288);
disp(length(x))
for k = 1:length(x)
    %disp(x(k))
    %disp(orig(k,1))
    %addpoints(h,x(k),orig(k,1));
    y_orig = orig(k,1);
    y_pred = pred(k,1);
    %disp(y)
addpoints(h,  x(k), y_orig);
addpoints(h2, x(k), y_pred);
%addpoints(h,x(k),pred(k,1));
drawnow
end
disp(orig(k,1))
