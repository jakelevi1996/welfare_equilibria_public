# welfare_equilibria_public

Set up with these commands:

```bash
python -m pip install -U pip
python -m pip install torch torchvision torchaudio numpy matplotlib jutility==0.0.14
```

Make figures from the paper with the following commands respectively:

```bash
# Figure 1
python scripts/plot_phase_portrait.py --game ImpossibleMarket  --a0_type Naive     --a1_type Naive     --learning_rate 0.01    --act_func linear   --xlim -2 2 --num_steps 100
python scripts/plot_phase_portrait.py --game ImpossibleMarket  --a0_type ExactLola --a1_type ExactLola --learning_rate 0.01    --act_func linear   --xlim -2 2 --num_steps 100  --a0_args 0.2  --a1_args 0.2
python scripts/plot_phase_portrait.py --game ImpossibleMarket  --a0_type Saga      --a1_type Saga      --learning_rate 0.01    --act_func linear   --xlim -2 2 --num_steps 500

# Figure 2
python scripts/plot_phase_portrait.py --game StagHunt          --a0_type Naive     --a1_type Naive     --learning_rate 0.1                                     --num_steps 50
python scripts/plot_phase_portrait.py --game StagHunt          --a0_type ExactLola --a1_type ExactLola --learning_rate 0.1     --a0_args 5     --a1_args 5     --num_steps 50
python scripts/plot_phase_portrait.py --game StagHunt          --a0_type Sasa      --a1_type Sasa      --learning_rate 0.1     --a0_args 50 5  --a1_args 50 5  --num_steps 25

# Figure 3
python scripts/train_marl.py --game IteratedPrisonersDilemma --a0_type Sasa        --a1_type Naive     --learning_rate 0.1 --a0_args 20 1 --num_steps 5000 --downsample_ratio 100

# Figure 4
python scripts/compare_welfuse.py

# Supplementary Material
python scripts/plot_1d_game.py --game MatchingPennies                                                    --no_title
python scripts/plot_1d_game.py --game ElusiveGame                                                        --no_title
python scripts/plot_1d_game.py --game AwkwardGame                                                        --no_title
python scripts/plot_1d_game.py --game StagHunt                                                           --no_title
python scripts/plot_1d_game.py --game ImpossibleMarket                           --xlim -2 2 --ylim -2 2 --no_title
python scripts/plot_1d_game.py --game IpdTftAlldMix                                                      --no_title
python scripts/plot_1d_game.py --game PrisonersDilemma                                                   --no_title
python scripts/plot_1d_game.py --game PrisonersDilemma   --welfare utilitarian                           --no_title
python scripts/plot_1d_game.py --game ChickenGame                                                        --no_title
python scripts/plot_1d_game.py --game ChickenGame        --welfare egalitarian                           --no_title
python scripts/plot_1d_game.py --game BabyChickenGame                                                    --no_title
python scripts/plot_1d_game.py --game BabyChickenGame    --welfare egalitarian                           --no_title
python scripts/plot_1d_game.py --game CoordinationGame                                                   --no_title
python scripts/plot_1d_game.py --game CoordinationGame   --welfare fairness                              --no_title
python scripts/plot_1d_game.py --game Tandem                                     --xlim -2 3 --ylim -2 3 --no_title
python scripts/plot_1d_game.py --game Tandem             --welfare egalitarian   --xlim -2 3 --ylim -2 3 --no_title
python scripts/plot_1d_game.py --game UltimatumGame                                                      --no_title
python scripts/plot_1d_game.py --game UltimatumGame      --welfare egalitarian                           --no_title
python scripts/plot_1d_game.py --game EagleGame                                                          --no_title
python scripts/plot_1d_game.py --game EagleGame          --welfare fairness                              --no_title
python scripts/plot_1d_game.py --game EagleGame          --welfare egalitarian                           --no_title
```
