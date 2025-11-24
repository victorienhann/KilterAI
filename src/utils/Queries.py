QUERIES = {
    "board_info" : """ 
               SELECT psls.layout_id, psls.set_id, ps.edge_left, ps.edge_right, 
                      ps.edge_bottom, ps.edge_top, psls.image_filename
               FROM product_sizes_layouts_sets AS psls
               JOIN product_sizes AS ps
               ON ps.id = psls.product_size_id 
               WHERE ps.name = $name AND ps.description = $description """,
    "holds" : """ 
                SELECT placements.id, mirrored_placements.id, holes.x, holes.y 
                FROM holes INNER JOIN placements 
                ON placements.hole_id = holes.id AND (placements.set_id = $set_id_1 OR placements.set_id = $set_id_2)  
                AND placements.layout_id = $layout_id LEFT JOIN placements mirrored_placements 
                ON mirrored_placements.hole_id = holes.mirrored_hole_id AND (mirrored_placements.set_id = $set_id_1 
                OR mirrored_placements.set_id = $set_id_2)  AND mirrored_placements.layout_id = $layout_id""",
    "edges" : """
                SELECT edge_left, edge_right, edge_bottom, edge_top
                FROM product_sizes 
                WHERE product_sizes.description = $name """,
    "board_climb" : """SELECT c.frames, c.angle, cs.display_difficulty
                   FROM  climbs c JOIN  climb_stats cs ON c.uuid = cs.climb_uuid 
                   WHERE c.angle IS NOT NULL AND c.layout_id = $layout_id AND c.edge_left >= $edge_left 
                    AND c.edge_right <= $edge_right AND c.edge_bottom >= $edge_bottom AND c.edge_top <= $edge_top 
                """,
    "images" : """SELECT psls.image_filename
               FROM product_sizes_layouts_sets AS psls
               JOIN product_sizes AS ps
               ON ps.id = psls.product_size_id 
               WHERE ps.name = $name AND ps.description = $description """
   }