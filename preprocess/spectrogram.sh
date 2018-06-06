for dir in ./genres/*; do
  cd $dir
  for filename in ./*.wav; do
    python ../../spectrogram.py "$filename"
  done
  cd ../..
done
