for ((i=1; i<=10; i=i+1))
do
 python panda_grasp_c&v.py --idx $i --num_envs 100 --gpu 1 \
 --num_objects 5 --obj_type 1 --headless 1
done

for ((i=1; i<=10; i=i+1))
do
 python panda_grasp_c&v.py --idx $i --num_envs 100 --gpu 1 \
 --num_objects 10 --obj_type 1 --headless 1
done

for ((i=1; i<=10; i=i+1))
do
 python panda_grasp_c&v.py --idx $i --num_envs 100 --gpu 1 \
 --num_objects 15 --obj_type 1 --headless 1
done

for ((i=1; i<=10; i=i+1))
do
 python panda_grasp_c&v.py --idx $i --num_envs 100 --gpu 1 \
 --num_objects 20 --obj_type 1 --headless 1
done