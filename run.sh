python3 manager.py -n 20 -p0 my_agent_basic.py -p1 my_agent_basic2.py -l base1_base2 >> base1_base2.txt

python3 manager.py -n 20 -p1 my_agent_basic.py -p0 my_agent_basic2.py -l base2_base1 >> base2_base1.txt

python3 manager.py -n 20 -p0 my_agent.py -p1 my_agent_basic.py -l agent_base >> agent_base.txt

python3 manager.py -n 20 -p1 my_agent.py -p0 my_agent_basic.py -l base_agent >> base_agent.txt

python3 manager.py -n 20 -p0 my_agent.py -p1 my_agent_basic2.py -l agent_base2 >> agent_base2.txt

python3 manager.py -n 20 -p1 my_agent.py -p0 my_agent_basic2.py -l base2_agent >> base2_agent.txt

python3 analyze_results.py