%% Function to check if a point is inside a rotated hexagon (90-degree rotation)
function inside = isInsideHexagon(i, j, centerX, centerY, hori, vert)
    % Rotate the coordinate system by 90 degrees
    q2x = abs(j - centerX);  % x-distance from the center becomes y
    q2y = abs(i - centerY);  % y-distance from the center becomes x

    % Bounding test: check if the point is outside the hexagon's bounding box
    if (q2x > hori || q2y > vert * 2)
        inside = false;
        return;
    end

    % Dot product test based on hexagonal symmetry
    inside = (2 * vert * hori - vert * q2x - hori * q2y) >= 0;
end