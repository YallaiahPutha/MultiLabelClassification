

def fsum(nums,target):
    ans=[]
    n=len(nums)
    tmp_left=0
    tmp_right=0
    nums.sort()
    if n>3:
        i=0
        j=0
        while i<n-3:
            j=i
            while j<n-2:
                l=j+1
                r=n-1
                while l<r:
                    s=nums[i]+nums[j]+nums[l]+nums[r]
                    if s>target:
                        r-=1
                    elif s<target:
                        l+=1
                    else:
                        lst=[]
                        tmp_left=nums[l]
                        tmp_right=nums[r]
                        lst.append(nums[i])
                        lst.append(nums[j])
                        lst.append(nums[l])
                        lst.append(nums[r])
                        ans.append(lst)
                        l+=1
                        r-=1
                        while l<r and nums[l]==tmp_left:
                            l+=1
                        while l<r and nums[r]==tmp_right:
                            r-=1
                while j<n-3 and nums[j]==nums[j+1]:
                    j+=1
                j+=1
            while i<n-4 and nums[i]==nums[i+1]:
                i+=1
            i+=1
    print(ans)




nums=[-2,-1,-1,1,1,2,2]
target=0
fsum(nums,target)