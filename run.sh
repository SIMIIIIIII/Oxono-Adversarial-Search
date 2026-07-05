python3 manager.py -n 50 -p0 my_agent_MCTS.py -p1 my_agent_MCTS_basic.py -l mcts_basic >> mcts_basic.txt

python3 manager.py -n 50 -p1 my_agent_MCTS.py -p0 my_agent_MCTS_basic.py -l basic_mcts >> basic_mcts.txt

python3 analyze_results.py mcts_basic.txt basic_mcts.txt  >> res_basic_mcts.txt
