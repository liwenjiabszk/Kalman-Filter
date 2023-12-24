function estimatedTrack = performEstimationEKF(observations, parameters)

% load system parameters
numSteps = parameters.numSteps;
scanTime = parameters.scanTime;
sigmaDrivingNoise = parameters.sigmaDrivingNoise;
sigmaMeasurementNoiseRange = parameters.sigmaMeasurementNoiseRange;
sigmaMeasurementNoiseBearing = parameters.sigmaMeasurementNoiseBearing;
sensorPosition = parameters.sensorPosition;
priorMean = parameters.priorMean;
priorCovariance = parameters.priorCovariance;

[A,W,~] = getModelMatrices(scanTime);
drivingNoiseCovariance = diag([sigmaDrivingNoise^2;sigmaDrivingNoise^2]); 
measurementNoiseCovariance = diag([sigmaMeasurementNoiseRange^2;sigmaMeasurementNoiseBearing^2]); 

% initialize estimated means and covariances
estimatedMeans = zeros(4,numSteps+1);
estimatedCovariances = zeros(4,4,numSteps+1);
estimatedMeans(:,1) = priorMean;
estimatedCovariances(:,:,1) = priorCovariance;


for step = 1:numSteps
    
    % Predict
    predictedState = A * estimatedMeans(:,step);
    predictedCovariance = A * estimatedCovariances(:,:,step) * A' + W * drivingNoiseCovariance * W';
    % EKF Update Step
    % Compute Kalman Gain
    H = calculateJacobian(predictedState, sensorPosition);
    S = H * predictedCovariance * H' + measurementNoiseCovariance;
    K = predictedCovariance * H' / S;

    % Update Step
    z = observations(:, step);
    z_predicted = getObservations(predictedState, parameters);
    z_pred = z_predicted(:, step);
    innovation = wrapToPi(z - z_pred);
    estimatedMeans(:,step+1) = predictedState + K * innovation;
    estimatedCovariances(:,:,step+1) = (eye(size(K,1)) - K * H) * predictedCovariance;

    estimatedCovariances(:,:,step+1) = checkAndFixCovarianceMatrix(estimatedCovariances(:,:,step+1), 10^(-10));

end

% remove prior information from estimated track
estimatedTrack = estimatedMeans(:,2:numSteps+1);
end

function H = calculateJacobian(predictedMean, sensorPosition)
    x_1n = predictedMean(1);
    x_2n = predictedMean(2);
    p1 = sensorPosition(1);
    p2 = sensorPosition(2);

    rangeSquared = (x_1n - p1)^2 + (x_2n - p2)^2;
    range = sqrt(rangeSquared);

    H11 = (x_1n - p1) / range;
    H12 = (x_2n - p2) / range;
    H21 = (x_2n - p2) / rangeSquared;
    H22 = (p1 - x_1n) / rangeSquared;

    H = [H11, H12, 0, 0;
         H21, H22, 0, 0];
end