import shutil, torch
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
save_checkpoint({
    'epoch': epoch + 1,
    'arch': args.arch,
    'state_dict': model.state_dict(),
    'best_prec1': best_prec1,
    'optimizer' : optimizer.state_dict(),
}, is_best)

'''
best_prec1 = <whatever>
while training:
    prec1 = validate(val_loader, model, criterion)

    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
'''
