%%1. Get data
% get NSynth test data, make metadata
if (~exist("nsynth-test", 'dir')) untar("nsynth-test.jsonwav.tar.gz"); end 
notes = struct2cell(jsondecode(fileread("nsynth-test/examples.json")));
[metadata, labels] = get_metadata(notes);


%%2. Generate figures
% histogram generation
for i = 1:numel(labels)
    label = labels{i};
    histogram(categorical(metadata.(label)));
    exportgraphics(gca, ['histogram_' label '.png']);

    % frequency data
    values = unique(metadata.(label));
    if (iscellstr(values)) values = string(values); end
    counts = [];
    for j = 1:numel(values)
        n =  numel(metadata(metadata.(label) == values(j), :).(label));
        counts = [counts n];
        disp(label + string(values(j))+": " + string(n));
    end
    disp([label;"min: " + string(min(counts));"max: " + string(max(counts))]);
end

%%3. Run YAMNet
% get YAMNet
downloadFolder = fullfile(tempdir,'YAMNetDownload');
loc = websave(downloadFolder,'https://ssd.mathworks.com/supportfiles/audio/yamnet.zip');
YAMNetLocation = tempdir;
unzip(loc,YAMNetLocation)
addpath(fullfile(YAMNetLocation,'yamnet'))
dirPath = "nsynth-test/audio";
ds = audioDatastore(dirPath, "IncludeSubfolders", false);

% Running the networks
for current_label= 1:numel(labels)

    % properly label the files
    ds.Labels = categorical(metadata.(labels{current_label}));
    ds = ds_label(ds, metadata, labels{current_label});

    % partitioning
    rng(493)
    [dsTrain, dsValidation, dsTest] = splitEachLabel(ds, 0.75, 0.15, 0.10, 'randomized');
    [trainFeatures, trainLabels] = featureLabelExtract(dsTrain);
    [validationFeatures, validationLabels] = featureLabelExtract(dsValidation);
    [testFeatures, testLabels] = featureLabelExtract(dsTest);

    % prepare model
    net = yamnet;
    uniqueLabels = categorical(unique(metadata.(labels{current_label})));
    numLabels = numel(uniqueLabels);
    lgraph = layerGraph(net.Layers);
    newDenseLayer = fullyConnectedLayer(numLabels,"Name","dense");
    lgraph = replaceLayer(lgraph,"dense",newDenseLayer);
    newClassificationLayer = classificationLayer("Name","Sound","Classes",uniqueLabels);
    lgraph = replaceLayer(lgraph,"Sound",newClassificationLayer);

    options = trainingOptions("adam", ...
    "ValidationData", {single(validationFeatures), validationLabels}, ...
    "Verbose", true, ...
    "MaxEpochs", 10, ... % this can be reduced to 5 or even to 3 if need be.
    "MiniBatchSize", 64, ...
    "OutputNetwork", "best-validation-loss", ...
    "Shuffle", "every-epoch", ...
    "InitialLearnRate", 0.0001, ...,
    "Plots", "training-progress", ...
    "ExecutionEnvironment", "gpu");

    % training
    net = trainNetwork(single(trainFeatures),trainLabels,lgraph,options);

    % evaluation
    preds = classify(net, testFeatures);
    acc = (nnz(testLabels == preds'))/numel(testLabels);
    disp([labels{current_label} ' performance on test data:']);
    disp(['accuracy: ' string(acc)]);
    confusionchart(testLabels, preds)
    exportgraphics(gca, "confusion_" + labels{current_label} + ".png");

    % remove temporary worksplace variables
    vars = {"lgraph" "preds" "dsTest" "dsTrain" "dsValidation" "options" "net" "acc"};
    clear(vars{:}); clear("vars");
end
% cleanup
clear

%%4. Functions
% function generateNames
% params: col - list of unique values for a feature
%         feat - name of the feature
% return: list of column names
% goal: names one-hot encoded/binary columns


function names = generateNames(col, feat)
    names = [];
    for i = 1:numel(col)
        names = [names feat + "_" + string(col(i))];
    end
end

% function featureLabelExtract
% params: an audioDatastore subset, adsSubset
% return: an array of features and associated labels for a given subset
% goal: makes feature and label cells from the adsSubset audioDatastore
%       through accessing each file one at a time
function [features, labels] = featureLabelExtract(adsSubset)
  features = [];
  labels = [];
  
  rng(5678)
  nums = randi(400, 5, 1);
  i = 1;
  prefix = adsSubset.Folders{:} + "\";
  suffix = "." + adsSubset.DefaultOutputFormat{:};
  while hasdata(adsSubset)
      [audioIn, fileInfo] = read(adsSubset);
      feat = yamnetPreprocess(audioIn, fileInfo.SampleRate);
      nSpectrums = size(feat, 4);
      features = cat(4, features, feat);
      labels = cat(2, labels, repelem(fileInfo.Label, 1, nSpectrums));

      % random spectrograms
      if (ismember(i, nums))
          imagesc(feat(:, :, randi(size(feat, 4))));
          xlabel("Mel band"); ylabel("Frame");
          colorbar; axis tight;
          exportgraphics(gca, "melspectrogram_" + fileInfo.FileName(1 + strlength(prefix):end - strlength(suffix)) + ".png");
      end
      i = i + 1;
  end
end


% function get_metadata(notes)
% params: notes - cell array
% return: metadata - cleaned data table of the argument cell array
%         labels - vector of labels that may be used (this is unsorted)
% goal: cleans data* in notes and returns the resulting table
%       *column removal, one-hot encoding
function [metadata, labels] = get_metadata(notes)
    qualities = ["bright" "dark" "distortion" "fast_decay" "long_release" "multiphonic" "nonlinear_env" "percussive" "reverb" "tempo-synced"];
    to_remove = ["instrument_source" "instrument_family" "qualities_str" "qualities" "instrument_str" "sample_rate" "instrument" "note"];
    q_names = generateNames(qualities, "quality");
    metadata = {};
    for i = 1:numel(notes)
        % add these features (row contents) through vertical concatenation
        n = struct2table(notes{i, :}, "AsArray", true);
        for j = 1:numel(qualities)
            n.(q_names(j)) = categorical(n.qualities{:}(j));
        end
        n = removevars(n, to_remove);
        % add row to table
        metadata = [metadata;n];
    end
    labels = metadata.Properties.VariableNames;
end


% function ds_label
% params: ds - datastore (image or audio)
%         metadata - table of metadata to be used for labeling
%         label_str - string of the label to use
% return: ds - updated datastore with proper labels
% goal: updates a datastore's Labels field by looking through the metadata
%       by note_str (which is also filename) to associate proper values to
%       the files
function ds = ds_label(ds, metadata, label_str)
    prefix = ds.Folders{:} + "\";
    suffix = "." + ds.DefaultOutputFormat{:};
    for i = 1:numel(ds.Files)

        % retrieve note_str from filename
        filename = ds.Files{i};
        h = filename(1 + strlength(prefix):end - strlength(suffix));

        % access the proper row in metadata using note_str as key
        label = metadata(metadata.note_str == string(h), :).(label_str);
        ds.Labels(i) = categorical(label);
    end
end
