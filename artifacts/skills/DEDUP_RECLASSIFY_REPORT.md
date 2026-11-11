# Dedup + reclassify (cycles 0–11 + v2_smoke)

- Raw simplified candidates: **127**
- After structural dedup: **120** (merged away 2 distinct-id rules; collapsed 5 same-id reprints)
- Status: `{'keep': 54, 'undercovered': 18, 'reject': 48}`
- Retained kinds: `{'uncertain': 2, 'basic_mechanics': 7, 'policy_skill': 29, 'basic_affordance': 8, 'basic_navigation': 15, 'progression_pattern': 11}`

## What counts as structural dedup (pipeline `deduplicate`)

### A. Same-id reprints across smoke/cycles (exact same rule, same hypothesis_id)
- `hyp_move_north_toward_tree_after_early_plant` ×2 from [v2_smoke, cycle9]
  - THEN `{"type": "action", "action": "Move North"}`
- `hyp_leave_southern_table_for_western_tree_when_wood_empty` ×2 from [v2_smoke, cycle9]
  - THEN `{"type": "action", "action": "Move West"}`
- `hyp_approach_eastern_skeleton_from_path_with_stone_sword` ×3 from [v2_smoke, cycle2, cycle9]
  - THEN `{"type": "action", "action": "Move East"}`
- `hyp_continue_stone_interaction_despite_northern_coal` ×2 from [cycle1, cycle9]
  - THEN `{"type": "action", "action": "Do"}`

### B. Distinct-id merges (exact match or broader condition subsumes narrower)
- **KEEP** `hyp_continue_tree_interaction_beside_zombie_without_stone_diversion` [mechanics_like]
  - THEN `{"type": "action", "action": "Do"}`
  - IF `{"field": "front", "op": "==", "value": "tree"}`
  - **MERGE AWAY** `hyp_continue_facing_tree_before_first_table_with_two_wood` (condition_subsumption_broader_kept)
    - away IF `{"all": [{"field": "front", "op": "==", "value": "tree"}, {"field": "is_night", "op": "==", "value": false}]}`
- **KEEP** `hyp_empty_wood_daytime_tree_target_before_first_wake` [teacher_candidate]
  - THEN `{"type": "next_interaction_target", "target": "tree"}`
  - IF `{"all": [{"field": "front", "op": "==", "value": "grass"}, {"field": "dist_tree", "op": "<=", "value": 2}]}`
  - **MERGE AWAY** `hyp_close_tree_target_over_eastern_stone_without_table` (condition_subsumption_broader_kept)
    - away IF `{"all": [{"field": "front", "op": "==", "value": "grass"}, {"field": "dir_stone", "op": "==", "value": "east"}, {"field": "visible_table", "op": "==", "value": false}, {"field": "has_wood_pickaxe", "op": "==", "value": true}, {"field": "has_wood_sword", "op": "==", "value": true}, {"field": "has_stone_pickaxe", "op": "==", "value": false}, {"field": "dist_tree", "op": "<=", "value": 2}, {"field": "inv_wood", "op": "<=", "value": 2}]}`

## Near-duplicate clusters among retained (NOT auto-dropped; same behavior + condition predicate subset)
- Clusters: **3**
### Cluster 1: suggest keep `hyp_mine_facing_stone_with_empty_stone_inventory_despite_distant_coal` [keep/uncertain]
- behavior: `{"type": "action", "action": "Do"}`
- `KEEP?` `hyp_mine_facing_stone_with_empty_stone_inventory_despite_distant_coal` [keep/uncertain] fid=0.603671706263499 support=12964 preds=1
- `dup?` `hyp_continue_stone_interaction_despite_northern_coal` [undercovered/basic_mechanics] fid=0.8695652173913043 support=23 preds=4
### Cluster 2: suggest keep `hyp_attack_cow_before_first_meal_with_existing_wood_sword_while_awake` [keep/policy_skill]
- behavior: `{"type": "action", "action": "Do"}`
- `KEEP?` `hyp_attack_cow_before_first_meal_with_existing_wood_sword_while_awake` [keep/policy_skill] fid=0.9257759784075573 support=741 preds=5
- `dup?` `hyp_interact_with_first_cow_without_sword_despite_high_food` [keep/policy_skill] fid=0.906345957011259 support=1954 preds=1
- `dup?` `hyp_feed_from_facing_cow_despite_thirst_with_stone_sword` [keep/policy_skill] fid=0.8883248730964467 support=394 preds=2
### Cluster 3: suggest keep `hyp_seek_northeastern_stone_after_cow_with_wood_reserve` [keep/basic_navigation]
- behavior: `{"type": "action_distribution", "actions": ["Move North", "Move East"]}`
- `KEEP?` `hyp_seek_northeastern_stone_after_cow_with_wood_reserve` [keep/basic_navigation] fid=0.6417910447761194 support=67 preds=5
- `dup?` `hyp_seek_northeastern_stone_with_four_wood_away_from_tree_and_cow` [keep/basic_navigation] fid=0.6 support=75 preds=4

## Trivial front==target (basic_affordance style)
- count: 8
- `hyp_replenish_empty_wood_from_tree_before_first_sword`
- `hyp_low_health_night_zombie_engagement_target`
- `hyp_night_facing_iron_target_after_stone_pickaxe_and_coal`
- `hyp_stone_target_beside_uncollected_iron_with_adequate_energy`
- `hyp_stone_target_before_first_collection_instead_of_sleep`
- `hyp_target_facing_water_before_first_drink_without_visible_enemies`
- `hyp_facing_western_stone_target_with_visible_diamond_before_upgrade`
- `hyp_continue_stone_target_despite_visible_diamond_before_furnace`
