# Crafter Behavior Skill

Use the most specific matching rule below. If no rule matches, fall back to the base policy. These rules describe action preferences only; they do not define whether an action is legal.

## Rules

1. When holding a sapling, before placing plant, before collecting wood, use `Place Plant`.
2. When holding exactly one sapling, before collecting wood, before placing plant, use `Place Plant`.
3. When facing skeleton, with no stone sword, use `Do`.
4. When facing zombie, use `Do`.
5. When facing cow, holding wood sword, holding exactly one wood, during the day, while awake, use `Do`.
6. When facing grass, holding wood sword, while awake, with cow to the South, with a cow exactly 2 steps away, with no tree visible, use `Move South`.
7. When facing cow, use `Do`.
8. When facing cow, holding stone sword, use `Do`.
9. When facing grass, holding wood sword, before collecting drink, while awake, with water to the West, with a water exactly 2 steps away, with no cow visible, with no stone visible, use `Move West`.
10. When before collecting wood, with no tree visible, with cow to the South-West, with no stone visible, with no sand visible, use `Move South`.
11. When facing path, holding at least 2 stone, holding at most 3 stone, with coal to the North, with a coal exactly 2 steps away, with no skeleton visible, use `Move North`.
12. When facing grass, holding wood sword, before eating cow, during the day, with no tree visible, with cow to the North-West, use one of `Move North`, `Move West`.
13. When facing path, holding exactly 4 stone, with no stone pickaxe, before placing furnace, with a table more than 1 step away, use `Place Furnace`.
14. When facing path, holding exactly 4 stone, holding coal, before placing furnace, with no zombie visible, with no skeleton visible, prioritize interacting with `furnace`.
15. When facing grass, holding a sapling, before placing plant, at night, with food ≥ 2, use `Place Plant`.
16. When facing path, holding exactly 4 stone, with no stone pickaxe, before placing furnace, with no zombie visible, with no skeleton visible, use `Place Furnace`.
17. When facing path, holding exactly one stone, with no coal, holding at least 1 wood, before placing stone, use `Place Stone`.
18. When no wood pickaxe, with tree to the South-East, with cow to the East, use one of `Move East`, `Move South`.
19. When facing grass, with a tree within 2 steps, prioritize interacting with `tree`.
20. When holding exactly 3 wood, with no wood pickaxe, use `Place Table`.
21. When facing grass, holding wood pickaxe, with no wood sword, holding wood, with a table within 1 step, use `Make Wood Sword`.
22. When facing tree, use `Do`.
23. When facing table, holding exactly one stone, holding wood, holding stone pickaxe, before crafting a stone sword, use `Make Stone Sword`.
24. When facing table, holding exactly one stone, holding exactly one wood, holding coal, before crafting a stone pickaxe, during the day, use `Make Stone Pickaxe`.
25. When facing table, holding coal, holding stone, holding wood, before collecting drink, before crafting a stone pickaxe, use `Make Stone Pickaxe`.
26. When facing path, before defeating skeleton, with skeleton to the West, with a skeleton within 2 steps, use `Move West`.
27. When facing path, before placing furnace, with skeleton to the West, with a skeleton exactly 2 steps away, use `Move West`.
28. When before collecting wood, before collecting a sapling, with tree to the South-West, use one of `Do`, `Move South`.
29. When facing stone, holding at least 2 wood, holding exactly one stone, with stone to the East, with health ≥ 8, use one of `Do`, `Move East`, `Make Stone Pickaxe`.
30. When facing stone, with stone to the North, with energy=9, use one of `Do`, `Move North`.
31. When facing grass, before collecting wood, before collecting a sapling, before eating cow, with tree to the South, with a tree exactly 2 steps away, use `Move South`.
32. When facing grass, with no wood pickaxe, with tree to the North, use `Move North`.
33. When facing path, with skeleton to the South, use `Move South`.
34. When facing grass, with no wood, with no wood pickaxe, after placing table, with tree to the South, use `Move South`.
35. When facing grass, with no wood sword, with tree to the North, use `Move North`.
36. When facing grass, holding wood pickaxe, before crafting a wood sword, with coal to the South, use `Move South`.
37. When facing grass, holding exactly 4 wood, with no stone, during the day, with stone to the North-East, use one of `Move North`, `Move East`.
38. When facing grass, holding wood sword, with no stone sword, before defeating zombie, while awake, with zombie to the North, use `Move North`.
39. When facing grass, holding exactly 4 wood, with no stone, with stone to the North-East, use one of `Move North`, `Move East`.
40. When facing stone, holding stone pickaxe, before placing furnace, with diamond visible, prioritize interacting with `stone`.
41. When facing zombie, prioritize interacting with `zombie`.
42. When facing tree, prioritize interacting with `tree`.
43. When facing stone, with energy=8, prioritize interacting with `stone`.
44. When facing stone, with energy ≥ 6, prioritize interacting with `stone`.
45. When facing water, before collecting drink, with no skeleton visible, prioritize interacting with `water`.
46. When facing stone, with no stone pickaxe, with stone to the West, with diamond visible, prioritize interacting with `stone`.
47. When facing iron, prioritize interacting with `iron`.
48. When facing plant, before collecting wood, with tree toward West, with a tree within 5 steps, pursue `collect_wood` as the next achievement.
49. When facing grass, holding a sapling, before placing plant, after defeating zombie, with a tree more than 2 steps away, pursue `place_plant` as the next achievement.
50. When facing grass, during the day, with a skeleton exactly 2 steps away, with a stone more than 2 steps away, pursue `defeat_skeleton` as the next achievement.
51. When facing path, holding exactly 4 stone, before placing furnace, pursue `place_furnace` as the next achievement.
52. When facing path, holding exactly one stone, with no stone pickaxe, before placing stone, before collecting coal, with a coal exactly 2 steps away, pursue `place_stone` as the next achievement.
53. When facing sand, holding wood pickaxe, before collecting drink, with water to the North, with a water exactly 2 steps away, with no skeleton visible, pursue `collect_drink` as the next achievement.
54. When facing stone, use `Do`.
55. When facing table, with no wood, with no wood pickaxe, with tree to the West, with table to the South, use `Move West`.
56. When facing furnace, holding stone pickaxe, before collecting drink, while awake, with water visible, with a water within 1 step, with drink ≤ 2, prioritize interacting with `water`.
57. When facing grass, holding wood pickaxe, with no wood, before collecting coal, while awake, with table to the North-East, with coal to the West, with a coal within 3 steps, use `Move West`.
58. When facing path, holding stone sword, with skeleton to the East, with a skeleton within 2 steps, with health ≥ 5, with no zombie visible, use `Move East`.
59. When facing grass, holding wood sword, before eating cow, while awake, with cow to the South, with no tree visible, with no stone visible, use `Move South`.
60. When facing grass, holding wood sword, while awake, with tree to the South-East, with cow to the West, with a cow within 2 steps, with a table more than 1 step away, use `Move West`.
61. When facing grass, before eating cow, with cow to the South, with no tree visible, with no stone visible, with energy=9, use `Move South`.
62. When facing grass, holding wood sword, before collecting drink, with water to the West, with a water within 2 steps, with no tree visible, with no stone visible, prioritize interacting with `water`.
63. When facing grass, holding wood pickaxe, before collecting stone, with a stone within 2 steps, with a tree more than 2 steps away, with a coal more than 2 steps away, with a cow more than 2 steps away, with a skeleton more than 2 steps away, prioritize interacting with `stone`.
64. When facing grass, holding exactly 2 wood, after collecting drink, after eating cow, with tree to the East, with a tree exactly 2 steps away, with no stone visible, prioritize interacting with `tree`.
65. When facing stone, holding exactly 2 stone, with no stone pickaxe, with coal to the North, use `Do`.
66. When facing grass, before collecting wood, after placing plant, with tree to the North, use `Move North`.
67. When holding exactly 4 stone, before placing furnace, with a tree more than 2 steps away, with a iron more than 2 steps away, with no skeleton visible, with a table more than 1 step away, pursue `place_furnace` as the next achievement.
68. When facing path, holding wood pickaxe, holding exactly 2 stone, before collecting coal, after placing stone, with a coal within 2 steps, with a stone more than 2 steps away, with no skeleton visible, pursue `collect_coal` as the next achievement.
69. When facing path, holding wood pickaxe, holding exactly 2 stone, before collecting coal, with a coal within 1 step, with a stone more than 1 step away, with a table more than 2 steps away, with no skeleton visible, pursue `collect_coal` as the next achievement.
70. When facing grass, holding wood sword, before collecting drink, with water to the West, with a water exactly 2 steps away, with no tree visible, with no stone visible, with no table visible, pursue `collect_drink` as the next achievement.
71. When facing grass, holding at least 6 wood, before collecting stone, with stone to the West, pursue `collect_stone` as the next achievement.
72. When facing grass, holding exactly 2 wood, with no wood pickaxe, before placing table, during the day, with tree to the South-East, with a tree at least 2 steps away, with cow to the North-East, use `Place Table`.
