def fun():
    r,c=2,2
    mat=[[2,3],[3,4]]
    m, n = len(mat), len(mat[0])
    if m * n != r * c:
        return mat

    ans = [[0] * c for _ in range(r)]
    for x in range(m * n):
        ans[x // c][x % c] = mat[x // n][x % n]

    return ans