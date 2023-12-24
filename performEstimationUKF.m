function estimatedTrack = performEstimationUKF(observations, parameters)

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
    %Kalman prediction step
    %epsilon = 1;
    predictedState = A * estimatedMeans(:,step);
    predictedCovariance = A * estimatedCovariances(:,:,step) * A' + W * drivingNoiseCovariance * W';

    % UKF update step
    [sigmaPoints, weights] = getSigmaPoints(predictedState, predictedCovariance);

    numSigmaPoints = size(sigmaPoints, 2);
    propagatedSigmaPoints = zeros(4, numSigmaPoints);
    qn = mvnrnd(zeros(2, 1), sigmaDrivingNoise^2 * eye(2));
    for i = 1:numSigmaPoints
        propagatedSigmaPoints(:, i) = A*sigmaPoints(:, i) + W * qn';
    end
    predictedStateSigma = sum(propagatedSigmaPoints*weights(:), 2);
    predictedCovarianceSigma = zeros(size(predictedCovariance));
    for i = 1:numSigmaPoints
        predictedCovarianceSigma = predictedCovarianceSigma + weights(i)*(sigmaPoints(:, i)-predictedStateSigma)*(sigmaPoints(:, i)-predictedStateSigma)';
    end
    predictedCovarianceSigma = predictedCovarianceSigma + W * drivingNoiseCovariance * W';

    z = zeros(2, numSigmaPoints);
    for i = 1:numSigmaPoints
        z(:, i) = observationModel(sigmaPoints(:, i),sensorPosition);
    end
    z_sigma = sum(z*weights(:), 2);

    Pzz = zeros(size(measurementNoiseCovariance));
    for i = 1:numSigmaPoints
        Pzz = Pzz + weights(i)*(z(:, i)-z_sigma)*(z(:, i)-z_sigma)';
    end
    Pzz = Pzz + measurementNoiseCovariance;

    Pxz = zeros(length(predictedStateSigma), length(z_sigma));
    for i = 1:numSigmaPoints
        Pxz = Pxz + weights(i)*(sigmaPoints(:, i)-predictedStateSigma)*(z(:, i)-z_sigma)';
    end
    
    % UKF: Calculate Kalman gain
    K = Pxz / Pzz;
    % UKF: Update step
    estimatedMeans(:, step + 1) = predictedStateSigma + K * (observations(:, step) - z_sigma);
    estimatedCovariances(:,:,step + 1) = predictedCovarianceSigma - K * Pzz * K';

    estimatedCovariances(:,:,step+1) = checkAndFixCovarianceMatrix( estimatedCovariances(:,:,step+1), 10^(-10) );
end

% remove prior information from estimated track
estimatedTrack = estimatedMeans(:,2:numSteps+1);
end