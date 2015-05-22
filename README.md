--------------------------------------------------------------------------------
Testing parsing alone:
--------------------------------------------------------------------------------
Usage: $ python test_parsing.py outdir inkmldir

test.py takes an output directory and a directory of inkml files to test.



--------------------------------------------------------------------------------
Testing segmentation, classification, parsing:
--------------------------------------------------------------------------------
Usage: $ python test_all.py stateFilename outdir inkmldir

test_all.py takes a filename for a serialized classifier, an output directory, and
a directory of inkml files to test.



--------------------------------------------------------------------------------
Testing segmentation and classification:
--------------------------------------------------------------------------------
Usage: $ python test.py stateFilename outdir (testFile.dat | inkmldir lgdir)

test.py takes a filename for a serialized classifier, an output directory, and
either a pickled testing data set or directories for inkml and lg files to test.
The lg files are used to ensure that the files created in the output directory
are as expected.


--------------------------------------------------------------------------------
Training
--------------------------------------------------------------------------------
Usage: $ python train.py (-nn|-rf|-et|modelFilename) outFilename
        (inFilename.dat | inkmldir lgdir)

train.py takes either a pickled, untrained classifier or an arguemet telling it
to create a 1-NN, random forest or extra trees cassifier with our set options.
It also takes an output filename (we've been using '.mdl' with these) and
either a pickled training data set or inkml and lg directories.

After it hasbeen run, a trained classifier while have been serialized to
outFilename.

--------------------------------------------------------------------------------
Data fold
--------------------------------------------------------------------------------
Usage: $ python split_inkmls.py (file.inkml | inkmlDirectory) lgdir trainingDir testingDir trainingLgDir testingLgDir [trainingPerc]]

OR

$ python split.py (file.inkml | inkmlDirectory) lgdir trainingFilename [testingFilename [trainingPerc]]

The former takes a directory full of inkml files (which may be in subfolders)
and a directory of lg files (which are assumed to contain at least those for
all the inkml files) and splits them up more or less evenly into seperate
directories which really ought to start out empty.

The latter instead serializes them to a pair of files, one for training, one for testing.