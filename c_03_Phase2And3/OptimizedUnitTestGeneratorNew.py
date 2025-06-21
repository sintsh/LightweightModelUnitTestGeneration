import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    PLBartForConditionalGeneration,
    PLBartTokenizer,
    get_scheduler,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    AdaptionPromptConfig,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import bitsandbytes as bnb
from datetime import datetime
import re
import signal
import sys

# Enable optimized settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

def setup_distributed():
    """Initialize distributed training"""
    try:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        print(f"Initialized process group: rank {local_rank} of {world_size}")
        return local_rank, world_size
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        raise

class OptimizedUnitTestGenerator:
    """Main class supporting PEFT methods for PLBART with distributed training"""
    
    PEFT_CONFIGS = {
        "lora": {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj"],  # PLBART specific
            "lora_dropout": 0.1,
            "bias": "none"
        },
        "qlora": {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj"],  # PLBART specific
            "lora_dropout": 0.1,
            "bias": "none",
            "quant_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        },
        "adapter": {
            "adapter_layers": 3,
            "adapter_len": 128
        }
    }

    def __init__(self, peft_type="lora", local_rank=0):
        self.local_rank = local_rank
        self.peft_type = peft_type.lower()
        self._validate_peft_type()
        self.timestamp = datetime.now().strftime("%Y%m%d")
        self.config = self._initialize_config()
        self._setup_directories()
        self.load_model_and_tokenizer()
        self.prepare_datasets()
        self.setup_training()

    def _validate_peft_type(self):
        """Validate the PEFT type"""
        if self.peft_type not in self.PEFT_CONFIGS:
            raise ValueError(f"Unsupported PEFT type: {self.peft_type}. "
                           f"Supported types: {list(self.PEFT_CONFIGS.keys())}")

    def _initialize_config(self):
        """Initialize configuration parameters"""
        base_config = {
            "model_name": "sintsh/AKPlbartT", # from hugging face
            "peft_type": self.peft_type,
            "learning_rate": 3e-5,
            "batch_size": 32,
            "max_length": 512,
            "num_epochs": 8,
            "early_stopping": 3,
            "gradient_accumulation": 2,
            "warmup_steps": 1000,
            "weight_decay": 0.01,
            "output_dir": f"./peft_models/{self.peft_type}_{self.timestamp}",
            "logging_dir": f"./peft_logs/{self.peft_type}_{self.timestamp}",
            "logging_steps": 500,
            "ddp_find_unused_parameters": False,
            "sample_size": 780944,
            "generation": {
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 2,
                "num_beams": 5,
                "max_new_tokens": 256
            }
        }
        return {**base_config, **self.PEFT_CONFIGS[self.peft_type]}

    def _setup_directories(self):
        """Create output directories"""
        if self.local_rank == 0:
            os.makedirs(self.config["logging_dir"], exist_ok=True)
            os.makedirs(self.config["output_dir"], exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.config["logging_dir"])
            print(f"TensorBoard logs will be saved to: {self.config['logging_dir']}")

    def _get_peft_config(self):
        """Get the appropriate PEFT configuration"""
        if self.peft_type in ["lora", "qlora"]:
            return LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=self.config["r"],
                lora_alpha=self.config["lora_alpha"],
                target_modules=self.config["target_modules"],
                lora_dropout=self.config["lora_dropout"],
                bias=self.config.get("bias", "none")
            )
        elif self.peft_type == "adapter":
            return AdaptionPromptConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                adapter_layers=self.config["adapter_layers"],
                adapter_len=self.config["adapter_len"]
            )
        else:
            raise ValueError(f"Unsupported PEFT type: {self.peft_type}")

    def load_model_and_tokenizer(self):
        """Initialize model with selected PEFT method"""
        if self.local_rank == 0:
            print(f"Initializing PLBART with {self.peft_type.upper()}...")

        model_args = {
            "pretrained_model_name_or_path": self.config["model_name"],
        }

        # Special handling for QLoRA
        if self.peft_type == "qlora":
            model_args.update({
                "quantization_config": self.config["quant_config"],
                "device_map": {"": f"cuda:{self.local_rank}"},
                "torch_dtype": torch.float16,
            })

            # Load model with quantization
            self.model = AutoModelForSeq2SeqLM.from_pretrained(**model_args)

            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Enable input require grads (critical for QLoRA)
            self.model.enable_input_require_grads()
            
            # Cast layer norms to fp32 for stability
            for name, module in self.model.named_modules():
                if "norm" in name:
                    module = module.to(torch.float32)

        else:
            # Standard LoRA or Adapter setup
            model_args["torch_dtype"] = torch.float16
            self.model = AutoModelForSeq2SeqLM.from_pretrained(**model_args)
            self.model = self.model.to(f"cuda:{self.local_rank}")

        # Apply PEFT configuration
        peft_config = self._get_peft_config()
        self.model = get_peft_model(self.model, peft_config)

        # Force enable gradients for all trainable parameters
        for param in self.model.parameters():
            if param.requires_grad:
                param.requires_grad_(True)  # Ensure gradients are enabled

        # Verify trainable parameters
        if self.local_rank == 0:
            self.model.print_trainable_parameters()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"Trainable parameter: {name}")

        # Special handling for QLoRA device placement
        if self.peft_type == "qlora":
            self.model.to(f"cuda:{self.local_rank}")
            
            # Configure gradient checkpointing properly
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        else:
            self.model = self.model.to(f"cuda:{self.local_rank}")

        # Wrap with DistributedDataParallel
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=self.config["ddp_find_unused_parameters"]
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])

        
    def prepare_datasets(self):
        """Prepare dataset with distributed sampler"""
        if self.local_rank == 0:
            print("Loading and preprocessing dataset...")
        
        # # full_dataset = load_dataset("method2test.csv")
        # full_dataset = load_dataset("csv", data_files="method2test.csv")
        # dataset = full_dataset['train'].shuffle(seed=42).select(
        #     range(min(self.config["sample_size"], len(full_dataset['train'])))
        # )

        
        # Load from Hugging Face dataset hub
        full_dataset = load_dataset("sintsh/m2test_dataset")

        # Shuffle and select sample size
        dataset = full_dataset['train'].shuffle(seed=42).select(
            range(min(self.config["sample_size"], len(full_dataset['train'])))
        )
        
        def preprocess_function(examples):
            inputs = self.tokenizer(
                examples["src_fm_fc_ms_ff"],
                max_length=self.config["max_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples["target"],
                    max_length=self.config["max_length"],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
            
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": labels["input_ids"].squeeze(0)
            }
        
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        self.train_dataset = split_dataset["train"].map(
            preprocess_function,
            batched=False,
            remove_columns=["src_fm_fc_ms_ff", "target"]
        )
        
        self.valid_dataset = split_dataset["test"].map(
            preprocess_function,
            batched=False,
            remove_columns=["src_fm_fc_ms_ff", "target"]
        )

    def setup_training(self):
        """Configure distributed training"""
        train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=int(os.environ.get("WORLD_SIZE", 1)),
            rank=self.local_rank
        )
        
        valid_sampler = DistributedSampler(
            self.valid_dataset,
            shuffle=False,
            num_replicas=int(os.environ.get("WORLD_SIZE", 1)),
            rank=self.local_rank
        )
        
        def collate_fn(batch):
            return {
                'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
                'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
                'labels': torch.stack([torch.tensor(item['labels']) for item in batch])
            }
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            sampler=train_sampler,
            pin_memory=True,
            num_workers=4,
            collate_fn=collate_fn
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config["batch_size"],
            sampler=valid_sampler,
            pin_memory=True,
            num_workers=4,
            collate_fn=collate_fn
        )
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = bnb.optim.AdamW8bit(
            params,
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        
        num_training_steps = len(self.train_loader) * self.config["num_epochs"]
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=num_training_steps
        )

    def train(self):
        """Distributed training loop with all PEFT types"""
        if self.local_rank == 0:
            print(f"Starting {self.config['peft_type'].upper()} training...")
            print(f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        scaler = torch.amp.GradScaler('cuda', enabled=(self.peft_type == "qlora"))
        best_loss = float('inf')
        patience = 0
        global_step = 0

        for epoch in range(self.config["num_epochs"]):
            self.model.train()
            self.train_loader.sampler.set_epoch(epoch)

            if self.local_rank == 0:
                progress_bar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}")

            total_loss = 0
            for step, batch in enumerate(self.train_loader):
                batch = {k: v.to(f"cuda:{self.local_rank}") for k, v in batch.items()}
                self.optimizer.zero_grad(set_to_none=True)

                try:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels']
                        )
                        loss = outputs.loss

                        # Debugging output
                        print(f"Loss: {loss.item()}, requires_grad: {loss.requires_grad}")

                        # Force gradient computation if necessary
                        if not loss.requires_grad:
                            loss = loss * 1.0  # Force gradient computation

                        total_loss += loss.detach().float()

                    if self.peft_type == "qlora":
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if (step + 1) % self.config["gradient_accumulation"] == 0:
                        if self.peft_type == "qlora":
                            scaler.unscale_(self.optimizer)

                        torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, self.model.parameters()),
                            max_norm=1.0
                        )

                        if self.peft_type == "qlora":
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            self.optimizer.step()

                        self.lr_scheduler.step()
                        global_step += 1

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"Rank {self.local_rank}: OOM at step {step}, skipping batch")
                        torch.cuda.empty_cache()
                        continue
                    raise

                if self.local_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item())

                    if global_step % self.config["logging_steps"] == 0:
                        self.writer.add_scalar("train/loss", loss.item(), global_step)
                        self.writer.add_scalar("train/learning_rate", 
                                            self.lr_scheduler.get_last_lr()[0], 
                                            global_step)

            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
            avg_train_loss = total_loss.item() / (len(self.train_loader) * dist.get_world_size())
            
            eval_loss = self.evaluate()

            if self.local_rank == 0:
                self.writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)
                self.writer.add_scalar("epoch/val_loss", eval_loss, epoch)
                self.writer.flush()

                print(f"\nEpoch {epoch + 1} Metrics:")
                print(f"Training Loss: {avg_train_loss:.4f}")
                print(f"Validation Loss: {eval_loss:.4f}")

                if eval_loss < best_loss:
                    best_loss = eval_loss
                    patience = 0
                    self.save_model()
                    print(f"Model improved - saved to {self.config['output_dir']}")
                else:
                    patience += 1
                    if patience >= self.config["early_stopping"]:
                        print("Early stopping triggered")
                        break

            torch.cuda.empty_cache()
                
    def evaluate(self):
        """Distributed validation"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.valid_loader:
                batch = {k: v.to(f"cuda:{self.local_rank}") for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                total_loss += outputs.loss.detach().float()
        
        torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        avg_loss = total_loss.item() / (len(self.valid_loader) * dist.get_world_size())
        
        if self.local_rank == 0:
            print(f"Validation Loss: {avg_loss:.4f}")
        
        return avg_loss

    def save_model(self):
        """Save model only from main process"""
        if self.local_rank != 0:
            return
            
        self.model.module.save_pretrained(self.config["output_dir"])
        self.tokenizer.save_pretrained(self.config["output_dir"])
        
        config = {
            "peft_type": self.config["peft_type"],
            "model_name": self.config["model_name"],
            "training_config": self.config
        }
        
        import json
        with open(os.path.join(self.config["output_dir"], "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def generate_test_case(self, java_code, num_retries=3):
        """Improved test case generation with error handling"""
        if self.local_rank != 0:
            return None
            
        prompt = java_code
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config["max_length"],
            truncation=True,
            padding="max_length"
        ).to(f"cuda:{self.local_rank}")
        
        best_output = None
        best_score = -1
        
        for attempt in range(num_retries):
            try:
                with torch.no_grad():
                    outputs = self.model.module.generate(
                        **inputs,
                        max_length=self.config["max_length"] + self.config["generation"]["max_new_tokens"],
                        num_beams=self.config["generation"]["num_beams"],
                        repetition_penalty=self.config["generation"]["repetition_penalty"],
                        no_repeat_ngram_size=self.config["generation"]["no_repeat_ngram_size"],
                        early_stopping=True
                    )
                    
                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    if "Generated JUnit test class:" in generated:
                        generated = generated.split("Generated JUnit test class:")[1].strip()
                    
                    cleaned = self.post_process_generation(generated)
                    score = self.validate_test_case(cleaned)
                    
                    if score > best_score:
                        best_output = cleaned
                        best_score = score
                        if score >= 2:
                            break
            
            except RuntimeError as e:
                print(f"Generation attempt {attempt+1} failed with: {str(e)}")
                if attempt == num_retries - 1:
                    return "Generation failed - please try again with different parameters"
                continue
        
        return best_output if best_output else self.get_fallback_test_case(java_code)

    def get_fallback_test_case(self, java_code):
        """Provide a basic template when generation fails"""
        class_match = re.search(r'class\s+(\w+)', java_code)
        class_name = class_match.group(1) if class_match else "TestClass"
        
        return f"""import org.junit.Test;
    import static org.junit.Assert.*;

    public class {class_name}Test {{
        @Test
        public void testMethod1() {{
            // TODO: Add test logic here
            assertTrue(true);
        }}
        
        @Test
        public void testMethod2() {{
            // TODO: Add test logic here
            assertEquals(expected, actual);
        }}
    }}"""

    def post_process_generation(self, generated_code):
        """Clean and format the generated test case"""
        generated_code = re.sub(r'```[a-z]*\n', '', generated_code)
        generated_code = generated_code.replace('```', '')
        
        imports = set(re.findall(r'^import\s+.*?;', generated_code, re.MULTILINE))
        
        class_match = re.search(r'public\s+class\s+(\w+Test)\s*\{[^}]*\}', generated_code, re.DOTALL)
        
        if class_match:
            test_class = class_match.group(0)
            
            if not any('org.junit' in imp for imp in imports):
                imports.add('import org.junit.Test;')
                imports.add('import static org.junit.Assert.*;')
            
            return '\n'.join(sorted(imports) + ['', test_class])
        
        test_methods = re.findall(
            r'(@Test\s+)?public\s+void\s+test\w+\s*\([^)]*\)\s*\{[^}]*\}', 
            generated_code
        )
        
        if test_methods:
            class_name = "GeneratedTest"
            if "class" in generated_code:
                class_match = re.search(r'class\s+(\w+)', generated_code)
                if class_match:
                    class_name = class_match.group(1) + "Test"
            
            standard_imports = {
                'import org.junit.Test;',
                'import static org.junit.Assert.*;'
            }
            all_imports = '\n'.join(sorted(standard_imports.union(imports)))
            
            return f"""{all_imports}

    public class {class_name} {{
        {''.join(test_methods)}
    }}"""
        
        return generated_code.strip()

    def validate_test_case(self, test_case):
        """Comprehensive validation of generated test case"""
        if not test_case.strip():
            return 0
            
        score = 0
        
        has_class = "class" in test_case and "{" in test_case
        has_method = re.search(r'public\s+void\s+test\w+\s*\(', test_case)
        has_assert = any(word in test_case.lower() 
                        for word in ["assert", "verify", "should", "expect"])
        has_test_annotation = "@Test" in test_case
        has_imports = "import org.junit" in test_case
        has_meaningful_name = bool(re.search(r'test\w+[A-Z][a-z]+', test_case))
        has_comments = "//" in test_case or "/*" in test_case
        
        if has_class: score += 1
        if has_method: score += 1
        if has_assert: score += 1
        if has_test_annotation: score += 1
        if has_imports: score += 1
        if has_meaningful_name: score += 1
        if has_comments: score += 0.5
        
        return min(int(score), 3)

    def cleanup(self):
        """Proper cleanup of distributed resources"""
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        if hasattr(self, 'writer') and self.local_rank == 0:
            self.writer.close()
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft_type", type=str, default="lora",
                       choices=["lora", "qlora", "adapter"])
    args = parser.parse_args()

    # Initialize distributed training
    local_rank, world_size = setup_distributed()

    if local_rank == 0:
        print(f"Starting distributed training on {world_size} GPUs")

    # Signal handling for graceful shutdown
    def handle_signal(signum, frame):
        print(f"Received signal {signum}, cleaning up...")
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    generator = None
    try:
        generator = OptimizedUnitTestGenerator(peft_type=args.peft_type, local_rank=local_rank)
        start_time = time.time()
        generator.train()

        if local_rank == 0:
            print(f"\nTraining completed in {(time.time()-start_time)/60:.2f} minutes")
            
            # Example test case generation
            java_code = """
            public class Calculator {
                public int add(int a, int b) { return a + b; }
                public boolean isEven(int n) { return n % 2 == 0; }
            }
            """
            print("\nGenerating test case...")
            test_case = generator.generate_test_case(java_code)
            print("\nGenerated Test Case:")
            print(test_case)

    except Exception as e:
        print(f"Error in main process (rank {local_rank}): {str(e)}", flush=True)
        raise
    finally:
        if generator:
            generator.cleanup()
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()





# Note: This code is designed to be run in a distributed environment with multiple GPUs.