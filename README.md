# nsynth-yamnet
Using MATLAB for transfer learning with pretrained YAMNet using the NSynth dataset

Also generates supplementary figures while preprocessing, training, and evaluating:
1. histograms for every label
2. 10 mel spectrogram images for each dataset partition for each label iterated
3. confusion matrices for every label
4. There is also a new window opened for each training progress on the network(s), which can be used to manually save accuracy and loss figures during training.

# Licensing
The dataset is made available by Google Inc. under a Creative Commons Attribution 4.0 International (CC BY 4.0)

[License](https://creativecommons.org/licenses/by/4.0/)

[License text](https://creativecommons.org/licenses/by/4.0/legalcode)

The source code (my source code) is licensed under the MIT License, in LICENSE.md

# Running the code:
## Dependencies:
This MATLAB live script requires the following toolboxes:

```
                Name                                Version     ProductNumber    Certain
    ___________________________________________    ________    _____________    _______

    {'MATLAB'                                 }    {'9.13'}           1          true  
    {'Deep Learning Toolbox'                  }    {'14.5'}          12          true  
    {'Statistics and Machine Learning Toolbox'}    {'12.4'}          19          true  
    {'Audio Toolbox'                          }    {'3.3' }         151          true  
```


Seeing the dependencies can be down by running the following code in the MATLAB command window:
```
>> [~,pList] = matlab.codetools.requiredFilesAndProducts('yamnet_nsynth.mlx');
>> squeeze(struct2table(pList))
```

This MATLAB live script also requires the NSynth musical note dataset, specifically the `json/wav` format of the test data, which can be found [here](https://magenta.tensorflow.org/datasets/nsynth#files).
Once downloaded, it ***must*** be in the same directory as this live script!

## Disclaimer
Running all of the code in the livescript may take a considerable amount of time, memory space (and the possibility of your computer to sound like a helicopter) due
to preprocessing, training, and evaluating the same network up to 14 times. 
To limit the number of loop iterations, breakpoints can be set at the 1st statement inside the body of the for loop at line 36.

Or to to limit the range of `current_label`. It can range from a value of 1 to 14.
The range can be changed in line 33:

`for current_label = 1:numel(labels) % numel(labels) is 14` to `for current_label = new_range_start:new_range_end`

The labels are as follows:

>1. pitch (128 classes)
>
>2. instrument_source_str (3 classes)
>
>3. velocity (5 classes)
>
>4. instrument_family_str (10 classes)
>
>5. quality_bright
>
>6. quality_dark
>
>7. quality_distortion
>
>8. quality_fast_decay
>
>9. quality_long_release
>
>10. quality_multiphonic
>
>11. quality_nonlinear_env
>
>12. quality_percussive
>
>13. quality_reverb
>
>14. quality_tempo-synced

Labels 5-14 are boolean classifiers.

