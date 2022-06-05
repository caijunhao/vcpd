for ((i=5; i<=20; i=i+5))
do
  for ((j=0; j<=2; j=j+1))
  do
    for ((k=1; k<=10; k=k+1))
    do
     python panda_grasp_cpn.py --headless 1 --num_envs 100 --gpu 1 \
     --num_objects $i --obj_type $j --idx $k
    done
  done
done

for ((i=5; i<=20; i=i+5))
do
  for ((j=0; j<=2; j=j+1))
  do
    for ((k=1; k<=10; k=k+1))
    do
     python panda_grasp_vgn.py --headless 1 --num_envs 100 --gpu 1 \
     --num_objects $i --obj_type $j --idx $k
    done
  done
done

for ((i=5; i<=20; i=i+5))
do
  for ((j=0; j<=2; j=j+1))
  do
    for ((k=1; k<=10; k=k+1))
    do
     python panda_grasp_vpn.py --headless 1 --num_envs 100 --gpu 1 --rn_used 0\
     --num_objects $i --obj_type $j --idx $k
    done
  done
done

for ((i=5; i<=20; i=i+5))
do
  for ((j=0; j<=2; j=j+1))
  do
    for ((k=1; k<=10; k=k+1))
    do
     python panda_grasp_vpn.py --headless 1 --num_envs 100 --gpu 1 --rn_used 1\
     --num_objects $i --obj_type $j --idx $k
    done
  done
done