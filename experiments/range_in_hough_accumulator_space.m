%%
xlow = 0;
xhigh = 902;
ylow = 0;
yhigh = 355;

theta_vals = linspace(0, pi, 1000);

figure(1);
clf;
plot(theta_vals, xlow * sin(theta_vals) + ylow * cos(theta_vals));
hold on;
plot(theta_vals, xhigh * sin(theta_vals) + ylow * cos(theta_vals));
plot(theta_vals, xlow * sin(theta_vals) + yhigh * cos(theta_vals));
plot(theta_vals, xhigh * sin(theta_vals) + yhigh * cos(theta_vals));

min_val = min([xlow * sin(theta_vals) + ylow * cos(theta_vals); ...
                xhigh * sin(theta_vals) + ylow * cos(theta_vals); ...
                xlow * sin(theta_vals) + yhigh * cos(theta_vals); ...
                xhigh * sin(theta_vals) + yhigh * cos(theta_vals)]);
            
max_val = max([xlow * sin(theta_vals) + ylow * cos(theta_vals); ...
    xhigh * sin(theta_vals) + ylow * cos(theta_vals); ...
    xlow * sin(theta_vals) + yhigh * cos(theta_vals); ...
    xhigh * sin(theta_vals) + yhigh * cos(theta_vals)]);

figure(2);
clf;
plot(theta_vals, log(1 ./ (max_val - min_val)));