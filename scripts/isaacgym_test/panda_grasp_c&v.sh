model=( cpn vpn rn )
for m in ${model[@]}
do
  for ((i=5; i<=20; i=i+5))
  do
    for ((j=0; j<=2; j=j+1))
    do
      for ((k=1; k<=10; k=k+1))
      do
       python panda_grasp_c\&v.py --headless 1 --num_envs 100 --gpu 1 \
       --num_objects $i --obj_type $j --idx $k --model_used $m
      done
    done
  done
done