function [estimatedTrack,innovationSequence] = performEstimationKalman(observations, parameters)

% load system parameters
numSteps = parameters.numSteps;
scanTime = parameters.scanTime;
sigmaDrivingNoise = parameters.sigmaDrivingNoise;
sigmaMeasurementNoise = parameters.sigmaMeasurementNoise;
priorMean = parameters.priorMean;
priorCovariance = parameters.priorCovariance;

[A,W,H] = getModelMatrices(scanTime);
drivingNoiseCovariance = diag([sigmaDrivingNoise^2;sigmaDrivingNoise^2]); 
measurementNoiseCovariance = diag([sigmaMeasurementNoise^2;sigmaMeasurementNoise^2]); 

% initialize estimated means and covariances
estimatedMeans = zeros(4,numSteps+1);
estimatedCovariances = zeros(4,4,numSteps+1);
estimatedMeans(:,1) = priorMean;
estimatedCovariances(:,:,1) = priorCovariance;
innovationSequence = zeros(2,numSteps);


for step = 1:numSteps
  %todo

    % Predict
    qn = mvnrnd(zeros(2, 1), sigmaDrivingNoise^2 * eye(2));
    predictedState = A * estimatedMeans(:,step) + W * qn.';
    predictedCovariance = A * estimatedCovariances(:,:,step) * A' + W * drivingNoiseCovariance * W';

    % Update
    innovation = observations(:,step) - H * predictedState;
    S = H * predictedCovariance * H' + measurementNoiseCovariance;
    K = predictedCovariance * H' / S;
    
    estimatedMeans(:,step+1) = predictedState + K * innovation;
    estimatedCovariances(:,:,step+1) = (eye(size(A)) - K * H) * predictedCovariance;

    innovationSequence(:,step) = innovation;

end

% remove prior information from estimated track
estimatedTrack = estimatedMeans(:,2:numSteps+1);
end