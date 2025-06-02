export PYTHONPATH=.

for i in {2..10}; do
  if [ $i -lt 10 ]; then
    track="track_0${i}.t"
  else
    track="track_${i}.t"
  fi
  python -m src.metaheuristic --track ${track} --iterations 20
done